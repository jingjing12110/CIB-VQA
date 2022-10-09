# @File :dib.py
# @Github :https://github.com/YannDubs/Mini_Decodable_Information_Bottleneck
import math
import operator
import numpy as np
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.distributions import Independent, Normal


# **************************************************************************
# Tools
# **************************************************************************
def freeze_(module):
    """Freezes a pytorch module."""
    for param in module.parameters():
        param.requires_grad = False
    module.eval()


def prod(iterable):
    """Take product of iterable like."""
    return reduce(operator.mul, iterable, 1)


def mean(l):
    """Take mean of array like."""
    return sum(l) / len(l)


class MLP(nn.Module):
    """Multi Layer Perceptron."""
    
    def __init__(self, dim_in, dim_out, n_hid_layers=1, dim_hid=128):
        super().__init__()
        # Leaky relu can help if deep
        layers = [nn.Linear(dim_in, dim_hid), nn.LeakyReLU()]
        for _ in range(1, n_hid_layers):
            layers += [nn.Linear(dim_hid, dim_hid), nn.LeakyReLU()]
        layers += [nn.Linear(dim_hid, dim_out)]
        self.module = nn.Sequential(*layers)
        self.initialize()
    
    def forward(self, x):
        return self.module(x)
    
    def initialize(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()
                nn.init.kaiming_uniform_(
                    m.weight, a=1e-2, nonlinearity="leaky_relu")


class ScaleGrad(Function):
    """Function which scales the gradients of the imputs by a scalar `lambd`."""
    
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.save_for_backward(x)
        ctx.lambd = lambd
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.needs_input_grad[0]:
            grad_output = grad_output * ctx.lambd
        return grad_output, None


scale_grad = ScaleGrad.apply


def MultivariateNormalDiag(loc, scale_diag):
    """Multi variate Gaussian with a diagonal covariance function."""
    if loc.dim() < 1:
        raise ValueError("loc must be at least one-dimensional.")
    return Independent(Normal(loc, scale_diag), 1)


def mean_logits(logits, dim=0):
    """Return the mean logit, where the average is taken over across
    `dim` in probability space."""
    log_prob = logits - torch.logsumexp(logits, -1, keepdim=True)
    log_mean_prob = torch.logsumexp(log_prob, 0) - math.log(log_prob.size(0))
    return log_mean_prob


class NaiveNuisanceGetter(nn.Module):
    """Class which return some random N in the y decomposition of X."""
    
    def __init__(self, n_heads, cardinality_X, cardinality_Y):
        super().__init__()
        nuisances = []
        for _ in range(n_heads):
            nuisances.append(
                torch.from_numpy(np.random.choice(cardinality_Y, cardinality_X))
            )
        self.register_buffer("nuisances", torch.stack(nuisances))
    
    def __call__(self, i, idcs):
        """Return the ith nuisance."""
        return self.nuisances[i, idcs]


class CrossEntropyLossGeneralize(nn.CrossEntropyLoss):
    """Cross entropy loss that forces (anti)-generalization.
    Note
    ----
    - we want to find an empirical risk minimizer that maximizes (antigeneralize) or minimizes
    (generalize) the test loss. Using a lagrangian relaxation of the problem this can be written
    as `min trainLoss + gamma * testLoss`, where the sign of `gamma` determines whether or not to
    generalize.
    - Each target should contain `(label, is_train)`. Where `is_train` says whether its a trainign
    example or a test example. `is_train=1` or `is_train=0`.
    Parameters
    ----------
    gamma : float, optional
        Langrangian coefficient of the relaxed problem. If positive, forces generalization, if negative
        forces anti generalization. Its scale balances the training and testing loss. If `gamma=0`
        becomes standard cross entropy.
    cap_test_loss : float, optional
        Value used to cap the test loss (i.e. don't backprop through it). This is especially useful
        when gamma is negative (anti generalization). Indeed, cross entropy is not bounded and thus
        the model could end up only focusing on maximizing the test loss to infinity regardless of
        train.
    is_forward_test : bool, optional
        Whether the forward pass should contain the loss from the test set or only from train.
        Useful for plotting, in case you only want to plot the training.
    kwargs :
        Additional arguments to `torch.nn.CrossEntropyLoss`.
    """
    
    def __init__(self, gamma=-0.1, cap_test_loss=10, is_forward_test=False,
                 **kwargs):
        super().__init__(reduction="none", **kwargs)
        self.gamma = gamma
        self.cap_test_loss = cap_test_loss
        self.is_forward_test = is_forward_test
    
    def forward(self, inp, targets):
        label, is_train = targets
        out = super().forward(inp, label)
        
        if self.gamma == 0:
            # simple cross entropy
            return out.mean()
        
        is_test = is_train == 0
        weights = is_test.int() * self.gamma + is_train.int()
        
        # CAPPING : don't backprop if test and larger than cap (but still forward)
        is_large_loss = out > self.cap_test_loss
        to_cap = is_large_loss & is_test
        out[to_cap] = out[to_cap] * 0 + out[to_cap].detach()
        
        out = weights * out
        
        if not self.is_forward_test:
            # set to 0 but still backprop
            out[is_test] = out[is_test] - out[is_test].detach()
            return out.sum() / is_train.sum()
        else:
            return out.mean()


# **************************************************************************
# Decodable Information Bottleneck
# **************************************************************************
class DIBWrapper(nn.Module):
    """Wrapper around encoders to make them usable with DIBLoss.
    Note
    ----
    This class can be used as a classifier with a DIB regularizer (1-player game setting) or as an
    encoder trainer with DIB (2-player game setting). In the former case, use directly the first
    output `y_pred` for predictions. In the latter case, once the first player trained the representation
    def `set_2nd_player()` to freeze the encoder and reset the classifier.
    Parameters
    ----------
    Encoder : nn.Module
        Core encoder. Initialized as `Encoder(x_shape, z_dim, **kwargs)`.
    V : nn.Module
        Desired functional family. Initialized as `V(z_dim, n_classes)`.
    x_shape : tuple, optional
        Size of the inputs.
    n_classes : int, optional
        Number of output classes.
    z_dim : int, optional
        Size of the representation.
    is_stochastic : bool, optional
        Whether to use a stochastic encoder.
    n_test_samples : int, optional
        Number of samples of z to use during testing if `is_stochastic`.
    is_contrain_norm : bool, optional
        Whether to ensure that the norm of `Z` cannot diverge. This is done by passing it through
        a batch normalization layer without parameters.
    kwargs :
        Additional arguments to `Encoder`.
    Return
    ------
    y_pred : torch.tensor, shape = [batch_size, n_classes]
    z_sample : torch.tensor, shape = [n_samples, batch_size, z_dim]
    """
    
    def __init__(
            self,
            Encoder,
            V,
            x_shape=(3, 32, 32),
            n_classes=10,
            z_dim=256,
            is_stochastic=True,
            n_test_samples=12,
            is_contrain_norm=True,
            **kwargs,
    ):
        
        super().__init__()
        self.n_classes = n_classes
        self.z_dim = z_dim
        self.is_stochastic = is_stochastic
        self.n_test_samples = n_test_samples
        self.is_contrain_norm = is_contrain_norm
        self.V = V
        
        # if stochastic doubles because will predict mean and variance
        enc_out_dim = z_dim * 2 if is_stochastic else z_dim
        self.encoder = Encoder(x_shape, enc_out_dim, **kwargs)
        
        if self.is_contrain_norm:
            self.batch_norm = torch.nn.BatchNorm1d(
                num_features=self.z_dim, affine=False
            )
        
        # V sufficiency head which can also be used for direct classification
        # in 1-player setting
        self.head_suff = V(z_dim, n_classes)
    
    def set_2nd_player_(self):
        """Set the model for the second stage by
        freezing encoder and resetting clf."""
        freeze_(self.encoder)
        self.head_suff = self.V(self.z_dim, self.n_classes)  # reset
    
    def forward(self, X):
        batch_size = X.size(0)
        
        # z_sample, shape=[n_samples, batch_size, z_dim]
        if self.is_stochastic:
            n_samples = 1 if self.training else self.n_test_samples
            z_suff_stat = self.encoder(X)
            z_mean, z_std = z_suff_stat.view(batch_size, -1, 2).unbind(-1)
            # use same initial std as in VIB to compare
            z_std = F.softplus(z_std - 5)
            p_zCx = MultivariateNormalDiag(z_mean, z_std)
            z_sample = p_zCx.rsample([n_samples])
            
            # DEV
            self.z_mean_norm = float(p_zCx.base_dist.loc.abs().mean())
            self.z_std = float(p_zCx.base_dist.scale.mean())
        else:
            n_samples = 1
            z_sample = self.encoder(X).unsqueeze(0)  # unsqueeze as if 1 sample
        
        if self.is_contrain_norm:
            z_sample = z_sample.view(batch_size * n_samples, -1)
            z_sample = self.batch_norm(z_sample)  # normalizaion over z_dim
            z_sample = z_sample.view(n_samples, batch_size, -1)
        
        # shape = [n_samples, batch_size, n_classes]
        y_preds = self.head_suff(z_sample)
        
        if n_samples == 1:
            y_pred = y_preds.squeeze(0)
        else:
            # take average prediction in proba space (slightly different than
            # in paper but more useful in DL application)
            y_pred = mean_logits(y_preds)
        
        return y_pred, z_sample


class DIBLoss(nn.Module):
    """DIB Loss.
    Note
    ----
    This is a simplification of the model we train in the paper. This version is simpler to code
    and more computationally efficient, but it performs a little worst than the model we use in the
    paper which is closer to the theory. Specifically the differences are:
        - I use joint optimization instead of unrolling optimization (see Appx. E.2)
        - I do not use y decompositions through base expansions (see Appx. E.5.)
        - I share predictors to improve batch training (see Appx. E.6.)
    Parameters
    ----------
    V : nn.Module
        Functional family for minimality.
    n_train : int
        Number of training examples.
    beta : float, optional
        Regularization weight.
    n_classes : int, optional
        Number of output classes.
    z_dim : int, optional
        Size of the representation.
    inp_min : {"Z", "Z,Y"}
        What input to use to the V-minimality heads. "Z" approximates V-minimality without considering
        the true label of the current example, i.e., it theoretically makes every example
        indistinguishable rather than only the ones with the same underlying label. In practice it
        works quite well.  "Z,Y" gives both the repsentation and $Y$ as input, to approximate "Z_y".
        In theory the functional family V is slightly differerent (as it takes Y as input), but
        that does not make much difference in practice. In the paper we use "Z_y" as input,which in
        practice it is slower as it is harder to parallelize (see Appx. E.6). To keep the code short
        I don't allow "Z_y" as input.
    n_heads : list, optional
        Number of V-minimality heads to use.
    """
    
    def __init__(
            self,
            V,
            n_train,
            beta=1,
            n_classes=10,
            z_dim=128,
            inp_min="Z,Y",
            n_heads=5,
    ):
        super().__init__()
        self.beta = beta
        self.n_classes = n_classes
        self.z_dim = z_dim
        self.inp_min = inp_min
        self.n_heads = n_heads
        # could improve results by using independent nuisances (using base Y
        # decompositions). if you want to decrease memory or don't know the size
        # of training data, can use hashes in which case you also
        # wouldn't need `n_train`
        self.Ni_getter = NaiveNuisanceGetter(
            self.n_heads, n_train, self.n_classes)
        self.sufficiency_loss = nn.CrossEntropyLoss()
        
        # V minimality heads
        # when if $Y$ given as input then increment inp_dim
        inp_dim = self.z_dim + int(self.inp_min == "Z,Y")
        self.heads_min = nn.ModuleList(
            [V(inp_dim, self.n_classes) for _ in range(self.n_heads)]
        )
    
    def get_sufficiency(self, y_pred, labels):
        """Compute V sufficiency loss: H_V[Y|Z]."""
        return F.cross_entropy(y_pred, labels)
    
    def get_minimality(self, z_sample, x_idcs, labels):
        """Compute V minimality loss: 1/k \sum_k H_V[N_k|inp_min]."""
        
        if self.inp_min == "Z,Y":
            # add the label Y as input to V minimality heads
            target = torch.repeat_interleave(
                labels.view(1, -1, 1).float(), z_sample.size(0), dim=0
            )
            z_sample = torch.cat([z_sample, target], dim=-1)
        
        # TODO: check if running asynchronously
        H_V_nCz = mean(
            [
                self.get_H_V_niCz(head, self.Ni_getter(i, x_idcs), z_sample)
                for i, head in enumerate(self.heads_min)
            ]
        )
        
        return H_V_nCz
    
    def get_H_V_niCz(self, head_min, n_i, z_sample):
        """Compute H_V[N_i|Z]"""
        marg_pred_ni = head_min(z_sample).squeeze(0)
        H_V_niCz = F.cross_entropy(marg_pred_ni, n_i)
        
        # negative Iv impossible (predicting worst than marginal) =>
        # predictions are maximally bad => don't backprop to encoder.
        # If wanted to be exact (we do that in paper) should compare to
        # prediction when using marginal P_y instead of log(n_classes)
        # which gives an upper bound on H[N_i]
        if H_V_niCz > math.log(self.n_classes):
            # only backprop through head rather than encoder
            # (recompute with detach Z)
            marg_pred_ni = head_min(z_sample.detach()).squeeze(0)
            H_V_niCz = F.cross_entropy(marg_pred_ni, n_i)
        
        return H_V_niCz
    
    def forward(self, out, targets):
        y_pred, z_sample = out
        labels, x_idcs = targets
        
        # V-Sufficiency ###
        H_V_yCz = self.get_sufficiency(y_pred, labels)
        
        # V-minimality ###
        # for the encoder: reverse gradients and * beta but not for the heads!!
        z_sample = scale_grad(z_sample, -self.beta)
        H_V_nCz = self.get_minimality(z_sample, x_idcs, labels)
        
        # DIB ###
        # no need of -self.beta due to gradient scaling
        dib = H_V_yCz + H_V_nCz
        # trick to still keep the dib loss correct for plotting
        dib = dib - H_V_nCz.detach() - self.beta * H_V_nCz.detach()
        
        # Plotting ###
        self.H_V_yCz = float(H_V_yCz)
        self.H_V_nCz = float(H_V_nCz)
        
        return dib

# if __name__ == '__main__':
#     V = SmallMLP  # architecture of the classifier
#     model = DIBWrapper(V=V, Encoder=LargeMLP)  # architecture of the encoder
#     loss = DIBLoss(V=V, n_train=50000)  # needs to know training size
#     train(model, loss, get_DIB_data(CIFAR10))
