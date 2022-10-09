# @File :mi_estimator.py
# @Time :2021/5/11
# @Desc :
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# *************************************************************************
# MINE: https://zhuanlan.zhihu.com/p/151256189
# *************************************************************************
class Mine(nn.Module):
    def __init__(self, input_size=2, hidden_size=100):
        super(Mine, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
        self.init_weights()
    
    def init_weights(self):
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.fc1.bias, val=0)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.fc2.bias, val=0)
        nn.init.normal_(self.fc3.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.fc3.bias, val=0)
    
    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        y = self.fc3(x)
        return y


def learn_mine(joint, marginal, ma_et, ma_rate=0.01):
    mine_net = Mine()
    # compute mutual information
    t = mine_net(joint)
    et = torch.exp(mine_net(marginal))
    mi_lb = torch.mean(t) - torch.log(torch.mean(et))
    
    # compute loss
    ma_et = (1 - ma_rate) * ma_et + ma_rate * torch.mean(et)
    loss = -(torch.mean(t) - (1 / ma_et.mean()).detach() * torch.mean(et))
    return mi_lb, loss, ma_et


class MINE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(MINE, self).__init__()
        self.T_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))
    
    def forward(self, x_samples, y_samples):
        # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()
        
        y_shuffle = y_samples[random_index]
        
        T0 = self.T_func(torch.cat([x_samples, y_samples], dim=-1))
        T1 = self.T_func(torch.cat([x_samples, y_shuffle], dim=-1))
        
        lower_bound = T0.mean() - torch.log(T1.exp().mean())
        
        # compute the negative loss (maximise loss == minimise -loss)
        return lower_bound
    
    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)

# **************************************
# JSD
# **************************************


# ********************************************************************
# InfoNCE
# ********************************************************************
class GlobalDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(128, 64, kernel_size=3)
        self.c1 = nn.Conv2d(64, 32, kernel_size=3)
        self.l0 = nn.Linear(32 * 22 * 22 + 64, 512)
        self.l1 = nn.Linear(512, 512)
        self.l2 = nn.Linear(512, 1)
    
    def forward(self, y, M):
        h = F.relu(self.c0(M))
        h = self.c1(h)
        h = h.view(y.shape[0], -1)
        h = torch.cat((y, h), dim=1)
        h = F.relu(self.l0(h))
        h = F.relu(self.l1(h))
        return self.l2(h)


class LocalDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(192, 512, kernel_size=1)
        self.c1 = nn.Conv2d(512, 512, kernel_size=1)
        self.c2 = nn.Conv2d(512, 1, kernel_size=1)
    
    def forward(self, x):
        h = F.relu(self.c0(x))
        h = F.relu(self.c1(h))
        return self.c2(h)


class PriorDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = nn.Linear(64, 1000)
        self.l1 = nn.Linear(1000, 200)
        self.l2 = nn.Linear(200, 1)
    
    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))


class DeepInfoMaxLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=1.0, gamma=0.1):
        super().__init__()
        self.global_d = GlobalDiscriminator()
        self.local_d = LocalDiscriminator()
        self.prior_d = PriorDiscriminator()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def forward(self, y, M, M_prime):
        # see appendix 1A of https://arxiv.org/pdf/1808.06670.pdf
        y_exp = y.unsqueeze(-1).unsqueeze(-1)
        y_exp = y_exp.expand(-1, -1, 26, 26)
        
        y_M = torch.cat((M, y_exp), dim=1)
        y_M_prime = torch.cat((M_prime, y_exp), dim=1)
        
        Ej = -F.softplus(-self.local_d(y_M)).mean()
        Em = F.softplus(self.local_d(y_M_prime)).mean()
        LOCAL = (Em - Ej) * self.beta
        
        Ej = -F.softplus(-self.global_d(y, M)).mean()
        Em = F.softplus(self.global_d(y, M_prime)).mean()
        GLOBAL = (Em - Ej) * self.alpha
        
        prior = torch.rand_like(y)
        
        term_a = torch.log(self.prior_d(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d(y)).mean()
        PRIOR = - (term_a + term_b) * self.gamma
        
        return LOCAL + GLOBAL + PRIOR


# *****************************************************************************
# MIGE: Mutual Information Gradient Estimation
# *****************************************************************************
def entropy_surrogate(estimator, samples):
    dlog_q = estimator.compute_gradients(samples.detach(), None)
    surrogate_cost = torch.mean(torch.sum(dlog_q.detach() * samples, -1))
    return surrogate_cost


class ScoreEstimator:
    def __init__(self):
        pass

    def rbf_kernel(self, x1, x2, kernel_width):
        return torch.exp(
            -torch.sum(torch.mul((x1 - x2), (x1 - x2)), dim=-1) / (
                    2 * torch.mul(kernel_width, kernel_width))
        )

    def gram(self, x1, x2, kernel_width):
        x_row = torch.unsqueeze(x1, -2)
        x_col = torch.unsqueeze(x2, -3)
        kernel_width = kernel_width[..., None, None]
        return self.rbf_kernel(x_row, x_col, kernel_width)

    def grad_gram(self, x1, x2, kernel_width):
        x_row = torch.unsqueeze(x1, -2)
        x_col = torch.unsqueeze(x2, -3)
        kernel_width = kernel_width[..., None, None]
        G = self.rbf_kernel(x_row, x_col, kernel_width)
        diff = (x_row - x_col) / (kernel_width[..., None] ** 2)
        G_expand = torch.unsqueeze(G, -1)
        grad_x2 = G_expand * diff
        grad_x1 = G_expand * (-diff)
        return G, grad_x1, grad_x2

    def heuristic_kernel_width(self, x_samples, x_basis):
        n_samples = x_samples.size()[-2]
        n_basis = x_basis.size()[-2]
        x_samples_expand = torch.unsqueeze(x_samples, -2)
        x_basis_expand = torch.unsqueeze(x_basis, -3)
        pairwise_dist = torch.sqrt(
            torch.sum(torch.mul(
                x_samples_expand - x_basis_expand,
                x_samples_expand - x_basis_expand), dim=-1)
        )
        k = n_samples * n_basis // 2
        top_k_values = torch.topk(torch.reshape(
            pairwise_dist, [-1, n_samples * n_basis]), k=k)[0]
        kernel_width = torch.reshape(
            top_k_values[:, -1], x_samples.size()[:-2])
        return kernel_width.detach()

    def compute_gradients(self, samples, x=None):
        raise NotImplementedError()
    
    
class SpectralScoreEstimator(ScoreEstimator):
    def __init__(self, n_eigen=None, eta=None, n_eigen_threshold=None):
        self._n_eigen = n_eigen
        self._eta = eta
        self._n_eigen_threshold = n_eigen_threshold
        super().__init__()

    def nystrom_ext(self, samples, x, eigen_vectors, eigen_values, kernel_width):
        M = torch.tensor(samples.size()[-2]).to(samples.device)
        Kxq = self.gram(x, samples, kernel_width)
        ret = torch.sqrt(M.float()) * torch.matmul(Kxq, eigen_vectors)
        ret *= 1. / torch.unsqueeze(eigen_values, dim=-2)
        return ret

    def compute_gradients(self, samples, x=None):
        if x is None:
            kernel_width = self.heuristic_kernel_width(samples, samples)
            x = samples
        else:
            _samples = torch.cat([samples, x], dim=-2)
            kernel_width = self.heuristic_kernel_width(_samples, _samples)

        M = samples.size()[-2]
        Kq, grad_K1, grad_K2 = self.grad_gram(samples, samples, kernel_width)
        if self._eta is not None:
            Kq += self._eta * torch.eye(M)

        eigen_values, eigen_vectors = torch.symeig(
            Kq, eigenvectors=True, upper=True)

        if (self._n_eigen is None) and (self._n_eigen_threshold is not None):
            eigen_arr = torch.mean(
                torch.reshape(eigen_values, [-1, M]), dim=0)

            eigen_arr = torch.flip(eigen_arr, [-1])
            eigen_arr /= torch.sum(eigen_arr)
            eigen_cum = torch.cumsum(eigen_arr, dim=-1)
            eigen_lt = torch.lt(eigen_cum, self._n_eigen_threshold)
            self._n_eigen = torch.sum(eigen_lt)
        
        if self._n_eigen is not None:
            eigen_values = eigen_values[..., -self._n_eigen:]
            eigen_vectors = eigen_vectors[..., -self._n_eigen:]
        
        eigen_ext = self.nystrom_ext(
            samples, x, eigen_vectors, eigen_values, kernel_width)
        grad_K1_avg = torch.mean(grad_K1, dim=-3)
        M = torch.tensor(M).to(samples.device)
        beta = -torch.sqrt(M.float()) * torch.matmul(
            torch.transpose(eigen_vectors, -1, -2),
            grad_K1_avg) / torch.unsqueeze(eigen_values, -1)
        grads = torch.matmul(eigen_ext, beta)
        self._n_eigen = None
        return grads


def MIGE(d, range_rho, num_sample, GenerateData, threshold=None, n_eigen=None):
    spectral_j = SpectralScoreEstimator(
        n_eigen=n_eigen, n_eigen_threshold=threshold)
    spectral_m = SpectralScoreEstimator(
        n_eigen=n_eigen, n_eigen_threshold=threshold)
    approximations = []
    for rho in range_rho:
        rho = torch.FloatTensor([rho]).cuda()
        rho.requires_grad = True
        xs_ys, xs, ys = GenerateData(d, rho, num_sample)

        ans = entropy_surrogate(spectral_j, xs_ys) - entropy_surrogate(
            spectral_m, ys)

        ans.backward()
        approximations.append(rho.grad.data)

    approximations = torch.stack(approximations).view(-1).detach().cpu().numpy()
    return approximations


class InfoNCE(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(InfoNCE, self).__init__()
        self.lower_size = 300
        self.F_func = nn.Sequential(
            nn.Linear(x_dim + y_dim, self.lower_size),
            nn.ReLU(),
            nn.Linear(self.lower_size, 1),
            nn.Softplus()
        )

    def forward(self, x_samples, y_samples):
        # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))
        # [s_size, 1]
        T0 = self.F_func(torch.cat([x_samples, y_samples], dim=-1))
        # [s_size, s_size, 1]
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim=-1))

        lower_bound = T0.mean() - T1.logsumexp(dim=1).mean() - np.log(sample_size)

        # compute the negative loss (maximise loss == minimise -loss)
        return lower_bound


class InfoNCEv2(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(InfoNCEv2, self).__init__()
        self.lower_size = 300
        self.F_func = nn.Sequential(
            nn.Linear(x_dim + y_dim, self.lower_size),
            nn.ReLU(),
            nn.Linear(self.lower_size, 1),
            nn.Softplus()
        )
    
    def forward(self, positive_local_x, positive_global_x,
                negative_local_x, negative_global_x, batch_size):
        # samples have shape [sample_size, dim]
        t0_list = []
        for i in range(0, len(positive_local_x), batch_size):
            local_batch = torch.stack(
                positive_local_x[i: i + batch_size])
            global_batch = torch.stack(
                positive_global_x[i: i + batch_size])
            # [s_size, 1]
            t0_list.append(self.F_func(torch.cat(
                [local_batch, global_batch], dim=-1))
            )
        
        t1_list = []
        for i in range(0, len(negative_local_x), batch_size):
            local_batch = torch.stack(
                negative_local_x[i: i + batch_size])
            global_batch = torch.stack(
                negative_global_x[i: i + batch_size])
            # [s_size, s_size, 1]
            sample_size = local_batch.shape[0]
            t1_list.append(self.F_func(torch.cat([
                local_batch.unsqueeze(0).repeat((sample_size, 1, 1)),
                global_batch.unsqueeze(1).repeat((1, sample_size, 1))], dim=-1))
            )
            # t1 = t1.logsumexp(dim=1).mean()

        t1_list = [a.logsumexp(dim=1).mean() for a in t1_list]
        lower_bound = torch.cat(t0_list, dim=0).mean() - (
            torch.stack(t1_list).mean())
        
        # compute the negative loss (maximise loss == minimise -loss)
        return -lower_bound

     
class CLUBv2(nn.Module):
    # CLUB: Mutual Information Contrastive Learning Upper Bound
    def __init__(self, x_dim, y_dim, beta=0):
        super(CLUBv2, self).__init__()
        self.hiddensize = y_dim
        self.version = 2
        self.beta = beta
    
    def mi_est_sample(self, x_samples, y_samples):
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()
        
        positive = torch.zeros_like(y_samples)
        negative = - (y_samples - y_samples[random_index]) ** 2 / 2.
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        return upper_bound
    
    def mi_est(self, x_samples, y_samples):  # [nsample, 1]
        positive = torch.zeros_like(y_samples)
        
        prediction_1 = y_samples.unsqueeze(1)  # [nsample, 1, dim]
        y_samples_1 = y_samples.unsqueeze(0)  # [1, nsample, dim]
        negative = - ((y_samples_1 - prediction_1) ** 2).mean(
            dim=1) / 2.  # [nsample, dim]
        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
    
    def loglikeli(self, x_samples, y_samples):
        return 0
    
    def update(self, x_samples, y_samples, steps=None):
        # no performance improvement, not enabled
        if steps:
            # beta anealing
            beta = self.beta if steps > 1000 else self.beta * steps / 1000
        else:
            beta = self.beta
        
        return self.mi_est_sample(x_samples, y_samples) * self.beta


class CLUB(nn.Module):
    # CLUB: Mutual Information Contrastive Learning Upper Bound
    """This class provides the CLUB estimation to I(X,Y)
    Method:
        forward(): provides the estimation with input samples
        loglikeli(): provides the log-likelihood of the approximation
            q(Y|X) with input samples
    Arguments:
        x_dim, y_dim : the dimensions of samples from X, Y respectively
        hidden_size : the dimension of the hidden layer of the approximation
            network q(Y|X)
        x_samples, y_samples : samples from X and Y, having shape
            [sample_size, x_dim/y_dim]
    """
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, y_dim),
                                      nn.Tanh())
    
    def get_mu_logvar(self, x_samples):
        return self.p_mu(x_samples), self.p_logvar(x_samples)
    
    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        
        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples) ** 2 / 2. / logvar.exp()
        
        prediction_1 = mu.unsqueeze(1)  # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)  # shape [1,nsample,dim]
        
        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1) ** 2).mean(
            dim=1) / 2. / logvar.exp()
        
        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
    
    def loglikeli(self, x_samples, y_samples):
        # unnormalized loglikelihood
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar
                ).sum(dim=1).mean(dim=0)
    
    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


class CLUBSample(nn.Module):
    # Sampled version of the CLUB estimator
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, y_dim))
        
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, y_dim),
                                      nn.Tanh())
    
    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
    
    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(
            dim=0)
    
    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        
        sample_size = x_samples.shape[0]
        # random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()
        
        positive = - (mu - y_samples) ** 2 / logvar.exp()
        negative = - (mu - y_samples[random_index]) ** 2 / logvar.exp()
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        return upper_bound / 2.
    
    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)

