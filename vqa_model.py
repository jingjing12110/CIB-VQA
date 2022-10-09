import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from param import parse_args
from .lxrt.entry import LXRTEncoderFeature, set_visual_config
from .lxrt.modeling import BertLayerNorm, GeLU
from .lxrt.modeling import LXMERTModelCIBv2
from module.ib_lib.mi import MVMIEstimator
from module.mi_lib.upper_bound import CLUB
from module.ib_lib.info_nce import info_nce

# Max length including <bos> and <eos>
MAX_VQA_LENGTH = 20
args = parse_args()


class VQAModel(nn.Module):
    """Original vqa implementation of lxmert"""
    
    def __init__(self, num_answers):
        super().__init__()
        
        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoderFeature(
            args,
            max_seq_length=MAX_VQA_LENGTH,
            mode='lx'
        )
        hid_dim = self.lxrt_encoder.dim
        
        # VQA Answer heads
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)
    
    def forward(self, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size
        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :return: (b, num_answer) The logit of each answers.
        """
        feat_seq, input_mask, x = self.lxrt_encoder(sent, (feat, pos))
        return {
            "logit": self.logit_fc(x),
            "pooled_output": x,
            "feat_seq": feat_seq,
            "input_mask": input_mask
        }


class CIBVQAModelV2(nn.Module):
    """VQA implementation of cross-modal Information bottleneck."""
    
    def __init__(self, num_answers, using_cib=False, mi_lb="JSD"):
        super().__init__()
        self.using_cib = using_cib
        self.mi_lb = mi_lb
        
        # Build LXRT encoder
        set_visual_config(args)
        self.lxrt_encoder = LXMERTModelCIBv2.from_pretrained(
            "bert-base-uncased",
        )
        self.hid_dim = 768
        
        # VQA Answer heads
        self.logit_fc = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim * 2),
            GeLU(),
            BertLayerNorm(self.hid_dim * 2, eps=1e-12),
            nn.Linear(self.hid_dim * 2, num_answers)
        )
        self.logit_fc.apply(self.lxrt_encoder.init_bert_weights)
        
        if using_cib:
            # self.mi_mode = "sample"
            # self.mi_upper_estimator = MIUpperBound()
            self.mi_ub_estimator = CLUB()
            if mi_lb in ["InfoNCE", "JSD", "NWJ", "MINE", "CPC"]:
                self.mi_lb_estimator = MVMIEstimator(lb_name=mi_lb)
                # self.mi_lb_estimator = InfoNCEv2(self.hid_dim, self.hid_dim)
            # elif mi_lb == "CPC":
            #     self.mi_lb_estimator = CPC(self.hid_dim, self.hid_dim)
            elif mi_lb == "FDV":
                from module.mi_lib.flo import BilinearFDVNCE
                self.mi_lb_estimator = BilinearFDVNCE()
            elif mi_lb == "FLO":
                from module.mi_lib.flo import BilinearFenchelInfoNCEOne
                self.mi_lb_estimator = BilinearFenchelInfoNCEOne()

    def forward(self, input_ids, segment_ids, input_mask,
                feat, pos, inputs_embeds=None, obj_embeds=None,
                beta=0.0005, alpha=0.005):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size
        """
        outputs = self.lxrt_encoder(
            input_ids=input_ids,
            token_type_ids=segment_ids,
            attention_mask=input_mask,
            inputs_embeds=inputs_embeds,
            visual_feats=(feat, pos),
            obj_embeds=obj_embeds,
            visual_attention_mask=None,
        )
        if self.using_cib and self.training:
            # ablated experiments
            # self.compute_ib(outputs, input_mask)
            
            outputs['mi_lg'] = 0.
            # self.max_cls_local_embed_dis(outputs, input_mask)
            # self.max_cls_local_embed(outputs, input_mask)
            # self.max_cls_local_embed_v2(outputs, input_mask)
            # MI between local-global representation
            outputs["mi_lg_loss_i"] = outputs['mi_lg'] * alpha
            
            # outputs["cib"] = 0
            if self.mi_lb in ["InfoNCE", "JSD", "NWJ", "MINE", "CPC"]:
                self.compute_cib_v2(outputs, input_mask)
            else:
                self.compute_cib(outputs, input_mask)
            # CIB constraint
            outputs["cib_loss_i"] = outputs['cib']
            
            loss_mi = outputs["mi_lg_loss_i"] + outputs["cib_loss_i"] * beta
            loss_mi.backward(retain_graph=True)
            
        outputs['logit'] = self.logit_fc(outputs['pooled_output'])
        # outputs.pop("l_embeds")
        # outputs.pop("v_embeds")
        # outputs.pop("pooled_output")
        
        return outputs
    
    def compute_ib(self, outputs, input_mask):
        lv_embeds = []
        for i, l in enumerate(input_mask.sum(1)):
            lv_embed = torch.cat([
                outputs["l_embeds"][-1][i][:l, :],
                outputs["v_embeds"][-1][i]],
                dim=0)
            lv_embeds.append(lv_embed)
        lv_embeds = torch.cat(lv_embeds, dim=0)
        
        outputs["cib"] = self.mi_ub_estimator(
            outputs["l_embeds"][0].view(
                -1, self.hid_dim)[input_mask.view(-1) > 0],
            lv_embeds
        )
        
    def compute_cib_v2(self, outputs, input_mask):
        outputs["upper_bound_l"] = self.mi_ub_estimator(
            outputs["l_embeds"][0].view(
                -1, self.hid_dim)[input_mask.view(-1) > 0],
            outputs["l_embeds"][-1].view(
                -1, self.hid_dim)[input_mask.view(-1) > 0]
        )
        outputs["upper_bound_v"] = self.mi_ub_estimator(
            outputs["v_embeds"][0].view(-1, self.hid_dim),
            outputs["v_embeds"][-1].view(-1, self.hid_dim)
        )
        # [CLS] token representation
        # cls_emb = outputs["l_embeds"][-1][:, 0, :]  # [bs, 768]
        # cls_emb = outputs["pooled_output"]
        l_embeds = []
        v_embeds = []
        l_embeds_2 = []
        v_embeds_2 = []
        for i in range(input_mask.shape[0]):
            l_embed = outputs["l_embeds"][-1][i][input_mask[i] > 0]
            v_embed = outputs["v_embeds"][-1][i]
            sim_score = F.normalize(l_embed, dim=-1) @ F.normalize(
                v_embed, dim=-1).transpose(0, 1)
            # for L
            l_embeds.append(l_embed)
            v_embeds.append(v_embed[sim_score.max(1)[1], :])
            # For V
            l_embeds_2.append(l_embed[sim_score.max(0)[1], :])
            v_embeds_2.append(v_embed)
        l_embeds = torch.cat(l_embeds + l_embeds_2, dim=0)
        v_embeds = torch.cat(v_embeds + v_embeds_2, dim=0)
        # l_embeds_2 = torch.cat(l_embeds_2, dim=0)
        # v_embeds_2 = torch.cat(v_embeds_2, dim=0)
        
        # lv, skl = self.mi_lb_estimator.forward_skl(
        #     l_embeds, v_embeds)
        outputs["lower_bound_lv"], outputs["skl"] = self.mi_lb_estimator(
            l_embeds, v_embeds)
        outputs["skl"] /= l_embeds.shape[0]
        
        # lv_2, skl_2 = self.mi_lb_estimator.forward_skl(
        #     l_embeds_2, v_embeds_2)
        # lv_2, skl_2 = self.mi_lb_estimator(l_embeds_2, v_embeds_2)
        # skl_2 /= v_embeds_2.shape[0]
        
        # outputs["lower_bound_lv"] = (lv + lv_2) / 2.0
        # outputs["skl"] = (skl + skl_2) / 2.0
        outputs["cib"] = outputs["upper_bound_l"] + outputs["upper_bound_v"] - \
            outputs["lower_bound_lv"] + outputs["skl"]
        # sim_score = F.normalize(
        #     outputs["v_embeds"][-1], dim=-1) @ F.normalize(
        #     cls_emb, dim=-1).unsqueeze(-1)
        # sim_idx = sim_score.squeeze().sort(dim=-1, descending=True)[1]
        # v_embeds = []
        # for i, l in enumerate(input_mask.sum(1)):
        #     v_embeds.append(outputs["v_embeds"][-1][i][sim_idx[i, :l], :])
        # v_embeds = torch.cat(v_embeds, dim=0)
        # outputs["lower_bound_lv"], outputs["skl"] = self.mi_lb_estimator(
        # #     l_embeds, v_embeds)
        # outputs["cib"] = outputs["upper_bound_l"] + outputs["upper_bound_v"] + \
        #     outputs["lower_bound_lv"] + outputs["skl"]
        
    def compute_cib(self, outputs, input_mask):
        outputs["upper_bound_l"] = self.mi_ub_estimator(
            outputs["l_embeds"][0].view(
                -1, self.hid_dim)[input_mask.view(-1) > 0],
            outputs["l_embeds"][-1].view(
                -1, self.hid_dim)[input_mask.view(-1) > 0]
        )
        outputs["upper_bound_v"] = self.mi_ub_estimator(
            outputs["v_embeds"][0].view(-1, self.hid_dim),
            outputs["v_embeds"][-1].view(-1, self.hid_dim)
        )

        # [CLS] token representation
        # cls_emb = outputs["l_embeds"][-1][:, 0, :]  # [bs, 768]
        # cls_emb = outputs["pooled_output"]
        l_embeds = []
        v_embeds = []
        l_embeds_2 = []
        v_embeds_2 = []
        for i in range(input_mask.shape[0]):
            l_embed = outputs["l_embeds"][-1][i][input_mask[i] > 0]
            v_embed = outputs["v_embeds"][-1][i]
            sim_score = F.normalize(l_embed, dim=-1) @ F.normalize(
                v_embed, dim=-1).transpose(0, 1)
            # for L
            l_embeds.append(l_embed)
            v_embeds.append(v_embed[sim_score.max(1)[1], :])
            # For V
            l_embeds_2.append(l_embed[sim_score.max(0)[1], :])
            v_embeds_2.append(v_embed)
        l_embeds = torch.cat(l_embeds, dim=0)
        v_embeds = torch.cat(v_embeds, dim=0)
        l_embeds_2 = torch.cat(l_embeds_2, dim=0)
        v_embeds_2 = torch.cat(v_embeds_2, dim=0)
        
        lv = self.mi_lb_estimator(l_embeds, v_embeds)
        skl = self.compute_skl(l_embeds, v_embeds)
        lv_2 = self.mi_lb_estimator(l_embeds_2, v_embeds_2)
        skl_2 = self.compute_skl(l_embeds_2, v_embeds_2)

        outputs["lower_bound_lv"] = (lv + lv_2) / 2.0
        outputs["skl"] = (skl + skl_2) / 2.0
        outputs["cib"] = outputs["upper_bound_l"] + outputs["upper_bound_v"] - \
            outputs["lower_bound_lv"] + outputs["skl"]
        # sim_score = F.normalize(
        #     outputs["v_embeds"][-1], dim=-1) @ F.normalize(
        #     cls_emb, dim=-1).unsqueeze(-1)
        # sim_idx = sim_score.squeeze().sort(dim=-1, descending=True)[1]
        # v_embeds = []
        # for i, l in enumerate(input_mask.sum(1)):
        #     v_embeds.append(outputs["v_embeds"][-1][i][sim_idx[i, :l], :])
        # v_embeds = torch.cat(v_embeds, dim=0)
        #
        # outputs["lower_bound_lv"] = self.mi_lb_estimator(l_embeds, v_embeds)
        # outputs["skl"] = self.compute_skl(l_embeds, v_embeds)
        #
        # outputs["cib"] = outputs["upper_bound_l"] + outputs["upper_bound_v"] - \
        #     outputs["lower_bound_lv"] + outputs["skl"]
    
    def max_cls_local_embed(self, outputs, input_mask):
        local_rep = []
        for i, l in enumerate(input_mask.sum(1)):
            lv_embed = torch.cat([
                # outputs["l_embeds"][-1][i][:l, :],
                outputs["l_embeds"][-1][i][1:l, :],
                outputs["v_embeds"][-1][i]],
                dim=0)
            local_rep.append(lv_embed)
        local_rep = torch.cat(local_rep, dim=0)
        # global_rep = outputs["pooled_output"]
        global_rep = outputs["l_embeds"][-1][:, 0, :]
        res = torch.mm(local_rep, global_rep.t())
        
        pos_mask, neg_mask = self.create_masks(input_mask.sum(1) + 35)
        num_nodes = pos_mask.size(0)
        num_graphs = pos_mask.size(1)
        
        # JSD
        e_pos = (math.log(2.) - F.softplus(- res * pos_mask)
                 ).sum() / num_nodes
        e_neg = res * neg_mask
        e_neg = (F.softplus(-e_neg) + e_neg - math.log(2.)).sum() / (
                num_nodes * (num_graphs - 1))
        # # DV
        # e_pos = (res * pos_mask).sum() / num_nodes
        # e_neg = res * neg_mask
        # x_max = e_neg.max(0)[0]
        # e_neg = ((((e_neg - x_max).exp()).sum(0)).log() + x_max - math.log(
        #     e_neg.size(0))).sum() / (num_nodes * (num_graphs - 1))
        # # RKL
        # e_pos = (-(-res * pos_mask).exp()).sum() / num_nodes
        # e_neg = res * neg_mask
        # e_neg = (e_neg - 1.).sum() / (num_nodes * (num_graphs - 1))
        
        outputs["mi_lg"] = e_neg - e_pos

    @staticmethod
    def max_cls_local_embed_v2(outputs, input_mask):
        local_rep, global_rep = [], []
        for i, l in enumerate(input_mask.sum(1)):
            lv_embed = torch.cat([
                # outputs["l_embeds"][-1][i][:l, :],
                outputs["l_embeds"][-1][i][1:l, :],
                outputs["v_embeds"][-1][i]],
                dim=0)
            local_rep.append(lv_embed)
            global_rep.append(outputs["pooled_output"][i].unsqueeze(0).repeat(
                l + 35, 1))
        global_rep = torch.cat(global_rep, dim=0)
        local_rep = torch.cat(local_rep, dim=0)
        outputs["mi_lg"] = info_nce(global_rep, local_rep)
        # mi_lg = []
        # for i, l in enumerate(input_mask.sum(1)):
        #     lv_embed = torch.cat([
        #         outputs["l_embeds"][-1][i][:l, :],
        #         # outputs["l_embeds"][-1][i][1:l, :],
        #         outputs["v_embeds"][-1][i]],
        #         dim=0)
        #     global_rep = outputs["pooled_output"][i].unsqueeze(0).repeat(
        #         l + 36, 1)
        #     mi_lg.append(info_nce(
        #         global_rep, lv_embed, reduction='sum'))
        # outputs["mi_lg"] = torch.stack(mi_lg, dim=0).mean()
        
    @staticmethod
    def max_cls_local_embed_dis(outputs, input_mask):
        local_rep = []
        for i, l in enumerate(input_mask.sum(1)):
            # lv_embed = torch.cat([
            #     outputs["l_embeds"][-1][i][:l, :],
            #     # outputs["l_embeds"][-1][i][1:l, :],
            #     outputs["v_embeds"][-1][i]],
            #     dim=0)
            local_rep.append(outputs["l_embeds"][-1][i][:l, :])
        # min_l = (input_mask.sum(1) + 36).min()
        min_l = (input_mask.sum(1)).min()
        local_rep = [a[:min_l] for a in local_rep]
        local_rep = torch.stack(local_rep, dim=0)
        # local_rep = pad_sequence(
        #     local_rep, batch_first=True, padding_value=0)
        res = 1 - F.cosine_similarity(
            local_rep, outputs["pooled_output"][..., None, :], dim=-1)
        outputs["mi_lg"] = res.sum(1).mean()
        # res = F.pairwise_distance(
        #     local_rep, outputs["pooled_output"][..., None, :])
        # outputs["mi_lg"] = res.mean()
        
    @staticmethod
    def create_masks(lens_a):
        pos_mask = torch.zeros((lens_a.sum(), len(lens_a))).cuda()
        neg_mask = torch.ones((lens_a.sum(), len(lens_a))).cuda()
        temp = 0
        for idx in range(len(lens_a)):
            for j in range(temp, lens_a[idx] + temp):
                pos_mask[j][idx] = 1.
                neg_mask[j][idx] = 0.
            temp += lens_a[idx]
        return pos_mask, neg_mask
        
    @staticmethod
    def compute_skl(zl, zv):
        kl_1_2 = F.kl_div(
            F.log_softmax(zl, dim=-1), F.softmax(zv, dim=-1),
            reduction='batchmean')
        kl_2_1 = F.kl_div(
            F.log_softmax(zv, dim=-1), F.softmax(zl, dim=-1),
            reduction='batchmean')
        return (kl_1_2 + kl_2_1).mean() / 2.
    
    def load(self, path):
        # Load state_dict from snapshot file
        print("Load LXMERT pre-trained model from %s" % path)
        state_dict = torch.load("%s_LXRT.pth" % path)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[len("module.bert."):]] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict
        
        # Print out the differences of pre-trained and model weights.
        load_keys = set(state_dict.keys())
        model_keys = set(self.lxrt_encoder.state_dict().keys())
        print()
        print("Weights in loaded but not in model:")
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Weights in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()
        
        # Load weights to model
        self.lxrt_encoder.load_state_dict(state_dict, strict=False)

