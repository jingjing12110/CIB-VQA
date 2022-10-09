# @File :cs_evaluate.py
# @Time :2021/6/5
# @Desc :
import os
import torch
from tqdm import tqdm

from utils import save_json
from data_loader.base_dataset import VQAEvaluator
from lxrt.tokenization import BertTokenizer
from trainer import DataTuple
from metric import calc_consistency


class ConsistencyEvaluation:
    def __init__(self, args, model):
        super(ConsistencyEvaluation, self).__init__()
        self.args = args
        
        # Using the bert tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )
        self.max_seq_length = 20
        
        self.model = model(num_answers=self.args.num_answer)
        self.model = self.model.cuda()
        # load weights
        self.load(self.args.load)
        self.model.eval()
        
        # load dataset
        self.data_tuple = None
        self.test_mode = None
        
        # ## preprocess data ##
        self.ori_rep_pairs = {}
        if self.args.cs_test == 'vqa_p2':
            self.ori_rep_pairs_syn = {}
            self.ori_rep_pairs_para = {}
            self.ori_rep_pairs_ant = {}
        
        self.result_score = {}  # A dict of VQA accuracies for all questions
        
        # evaluate consistency
        self.evaluate_consistency()
    
    def evaluate_consistency(self):
        if self.args.cs_test == 'vqa_rep':
            # load dataset
            self.data_tuple = self.load_data()
            # ## preprocess data ##
            self.generate_ori_rep_pair()
            # compute vqa score
            self.result_score = self.predict(self.data_tuple)
            # compute cs
            cs_result = self.compute_cs(k_val=(1, 2, 3, 4))
            
            delta = float(cs_result['original acc']) - float(
                cs_result['rephrase acc'])
            cs_result['delta'] = f"{delta:.2f}"
            
            print(f"{self.args.cs_test}:")
            print(cs_result)
            print(f"Delta (original - rephrase): {delta:.2f}")
            
            # save results
            save_json(cs_result, os.path.join(
                self.args.load[:-4], f'{self.args.cs_test}_cs_result.json'))
        elif self.args.cs_test == 'vqa_p2':
            # load dataset
            self.data_tuple = self.load_data()
            # ## preprocess data ##
            self.generate_ori_rep_pair()
            # compute vqa score
            self.result_score = self.predict(self.data_tuple)
            # cs for all
            cs_result = self.compute_cs(k_val=(1, 2))
            # cs for synonymous
            self.ori_rep_pairs = self.ori_rep_pairs_syn
            cs_result_syn = self.compute_cs(k_val=(1, 2))
            # cs for paraphrastic
            self.ori_rep_pairs = self.ori_rep_pairs_para
            cs_result_para = self.compute_cs(k_val=(1, 2))
            # cs for synonymous
            self.ori_rep_pairs = self.ori_rep_pairs_ant
            cs_result_ant = self.compute_cs(k_val=(1, 2))
            
            delta = float(cs_result['original acc']) - float(
                cs_result['rephrase acc'])
            
            cs_result = {
                "cs_result_all": cs_result,
                "cs_result_syn": cs_result_syn,
                "cs_result_para": cs_result_para,
                "cs_result_ant": cs_result_ant,
                "delta": f"{delta:.2f}"
            }
            print(f"{self.args.cs_test}:")
            print(cs_result)
            print(f"Delta (original - rephrase): {delta:.2f}")
            
            save_json(cs_result, os.path.join(
                self.args.load[:-4], f'{self.args.cs_test}_cs_result_all.json'))
        elif self.args.cs_test == 'vqa_ce':
            # load dataset
            self.test_mode = 'all'
            self.data_tuple = self.load_data()
            # compute vqa score
            self.result_score = self.predict(self.data_tuple)
            overall_acc = sum(self.result_score.values()) / len(self.result_score)
            
            # counterexample
            self.test_mode = 'counterexample'
            self.data_tuple = self.load_data()
            # compute vqa score
            self.result_score = self.predict(self.data_tuple)
            counter_acc = sum(self.result_score.values()) / len(self.result_score)
            
            # easy
            self.test_mode = 'easy'
            self.data_tuple = self.load_data()
            # compute vqa score
            self.result_score = self.predict(self.data_tuple)
            easy_acc = sum(self.result_score.values()) / len(self.result_score)
            
            # hard
            self.test_mode = 'hard'
            self.data_tuple = self.load_data()
            # compute vqa score
            self.result_score = self.predict(self.data_tuple)
            hard_acc = sum(self.result_score.values()) / len(self.result_score)
            
            acc_result = {
                "overall acc": f"{100. * overall_acc:.2f}",
                "counterexample acc": f"{100. * counter_acc:.2f}",
                "easy acc": f"{100. * easy_acc:.2f}",
                "hard acc": f"{100. * hard_acc:.2f}",
                "delta": f"{100. * (easy_acc - counter_acc):.2f}"
            }
            print(f"{self.args.cs_test}: \n"
                  f"{acc_result}")
            print(f"Delta (easy - counterexample): "
                  f"{100. * (easy_acc - counter_acc):.2f}")
            
            save_json(acc_result, os.path.join(
                self.args.load[:-4], f'{self.args.cs_test}_acc_result.json'))
        elif self.args.cs_test in ['iv_vqa', 'cv_vqa']:
            # load dataset
            self.test_mode = 'edited'
            self.data_tuple = self.load_data()
            edited_result_score = self.predict(self.data_tuple)
            
            self.test_mode = 'original'
            data_tuple = self.load_data()
            original_result_score = self.predict(data_tuple)
            
            acc_result = self.compute_flips(
                edited_result_score, original_result_score)
            
            all_score = {}
            all_score.update(edited_result_score)
            all_score.update(original_result_score)
            acc_result['all_acc'] = \
                f"{100. * sum(all_score.values()) / len(all_score):.2f}"
            
            delta = float(acc_result['original_acc']) - float(
                acc_result['edited_acc'])
            acc_result['delta'] = f"{delta:.2f}"
            
            print(f"{self.args.cs_test}: \n"
                  f"{acc_result}")
            print(f"Delta (original - edited): {delta:.2f}")
            
            save_json(acc_result, os.path.join(
                self.args.load[:-4], f'{self.args.cs_test}_acc_result.json'))
        else:
            raise ValueError("please choose correct evaluation dataset.")
    
    def generate_ori_rep_pair(self):
        if self.args.cs_test == 'vqa_rep':
            img2ann = {}
            for ann in self.data_tuple.dataset.data:
                img_id = f"{ann['image_id']}"
                if img_id in img2ann.keys():
                    img2ann[img_id].append(ann)
                else:
                    img2ann[img_id] = [ann]
            # compute {ori_qid: [rep_qid, ...]}
            img_ids = [a['image_id'] for a in self.data_tuple.dataset.data]
            img_ids = list(set(img_ids))
            for img_id in img_ids:
                ori, rep = [], []
                for ann in img2ann[f"{img_id}"]:
                    if 'rephrasing_of' in ann.keys():
                        rep.append(ann['question_id'])
                    else:
                        ori = ann['question_id']
                self.ori_rep_pairs[f"{ori}"] = rep
        elif self.args.cs_test == 'vqa_p2':
            for ann in self.data_tuple.dataset.data_p2:
                ori = f"{ann['original_id']}"
                rep = f"{ann['question_id']}"
                if f"{ori}" in self.ori_rep_pairs.keys():
                    self.ori_rep_pairs[ori].append(rep)
                else:
                    self.ori_rep_pairs[ori] = [rep]
                perturbation = ann['perturbation']
                if perturbation == 'syn':
                    if f"{ori}" in self.ori_rep_pairs_syn.keys():
                        self.ori_rep_pairs_syn[ori].append(rep)
                    else:
                        self.ori_rep_pairs_syn[ori] = [rep]
                elif perturbation == 'para':
                    if f"{ori}" in self.ori_rep_pairs_para.keys():
                        self.ori_rep_pairs_para[ori].append(rep)
                    else:
                        self.ori_rep_pairs_para[ori] = [rep]
                elif perturbation == 'ant':
                    if f"{ori}" in self.ori_rep_pairs_ant.keys():
                        self.ori_rep_pairs_ant[ori].append(rep)
                    else:
                        self.ori_rep_pairs_ant[ori] = [rep]
        else:
            raise ValueError('cs_test should be vqa_rep or vqa_p2.')
        print(f"generating ori-rep pairs: {len(self.ori_rep_pairs)}")
    
    def compute_cs(self, k_val=(1, 2, 3, 4)):
        """compute Consensus Scores
        :param k_val: k from the CS definition
        """
        all_k_total_cons = {}
        # all_k_results_cons = {}
        num_groupings = len(self.ori_rep_pairs)
        num_ori_questions = 0
        num_rep_questions = 0
        ori_acc, rep_acc, all_acc = 0., 0., 0.
        
        if isinstance(self.ori_rep_pairs, dict):
            groupings_list = list(self.ori_rep_pairs.items())
        else:
            groupings_list = self.ori_rep_pairs
        
        for i, csk in enumerate(k_val):
            total_cons = 0.
            results_cons = []
            
            for ori_idx, rep_indices in groupings_list:
                group_ori_acc = [self.result_score[ori_idx]]
                group_rep_acc = [self.result_score[qid] for qid in rep_indices]
                group_acc = group_ori_acc + group_rep_acc
                group_cons = calc_consistency(group_acc, csk)
                total_cons += group_cons
                results_cons.append(group_cons)
                
                if i == 0:
                    ori_acc += sum(group_ori_acc)
                    num_ori_questions += len(group_ori_acc)
                    
                    rep_acc += sum(group_rep_acc)
                    num_rep_questions += len(group_rep_acc)
                    
                    all_acc += sum(group_acc)
            
            all_k_total_cons[csk] = 100. * (total_cons / num_groupings)
            # all_k_results_cons[csk] = results_cons
        
        ori_acc /= num_ori_questions
        rep_acc /= num_rep_questions
        all_acc /= (num_ori_questions + num_rep_questions)
        
        # mean_pert_acc = (ori_acc + rep_acc) / 2.
        
        cs_result = {
            "overall acc": f"{100. * all_acc:.2f}",
            "original acc": f"{100. * ori_acc:.2f}",
            "rephrase acc": f"{100. * rep_acc:.2f}",
            "cs": all_k_total_cons,
        }
        return cs_result
    
    def compute_flips(self, edited_result_score, original_result_score):
        edited_acc = sum(
            edited_result_score.values()) / len(edited_result_score)
        original_acc = sum(
            original_result_score.values()) / len(original_result_score)
        
        edited_org_qid_pairs = [
            (ann['question_id'], ann['question_id'].split('-')[0])
            for ann in self.data_tuple.dataset.data]
        n2p = []
        p2n = []
        n2n = []
        for pair in edited_org_qid_pairs:
            score_edited = edited_result_score[pair[0]]
            score_original = original_result_score[pair[1]]
            if score_original == 0. and score_edited > 0.:
                n2p.append(pair)
            elif score_original > 0. and score_edited == 0.:
                p2n.append(pair)
            elif score_original == 0. and score_edited == 0.:
                n2n.append(pair)
        flips = n2p + p2n + n2n
        
        acc_result = {
            "edited_acc": f"{100. * edited_acc:.2f}",
            "original_acc": f"{100. * original_acc:.2f}",
            "flips": f"{100. * (len(flips) / len(edited_org_qid_pairs)):.2f}",
            "n2p": f"{100. * (len(n2p) / len(edited_org_qid_pairs)):.2f}",
            "p2n": f"{100. * (len(p2n) / len(edited_org_qid_pairs)):.2f}",
            "n2n": f"{100. * (len(n2n) / len(edited_org_qid_pairs)):.2f}"
        }
        return acc_result
    
    def predict(self, eval_tuple: DataTuple):
        qid2ans = {}
        result_score = {}
        with torch.no_grad():
            dataset, loader, evaluator = eval_tuple
            tbar = tqdm(total=len(loader), ascii=True, ncols=80)
            for i, datum_tuple in enumerate(loader):
                # Avoid seeing ground truth
                ques_id, feats, boxes, sents, target = datum_tuple[:5]
                feats, boxes = feats.cuda(), boxes.cuda()
                target = target.cuda()
                
                # baseline LXMERT
                outputs = self.model(
                    feats, boxes, sents,
                )
                
                # CIB-based LXMERT
                # train_features = convert_sents_to_features(
                #     sents, self.max_seq_length, self.tokenizer)
                # input_ids = torch.tensor([f.input_ids for f in train_features],
                #                          dtype=torch.long).cuda()
                # input_mask = torch.tensor([f.input_mask for f in train_features],
                #                           dtype=torch.long).cuda()
                # segment_ids = torch.tensor(
                #     [f.segment_ids for f in train_features],
                #     dtype=torch.long).cuda()
                # outputs = self.model(
                #     input_ids=input_ids,
                #     segment_ids=segment_ids,
                #     input_mask=input_mask,
                #     feat=feats, pos=boxes,
                # )
                
                for idx, (qid, l) in enumerate(zip(
                        ques_id, outputs['logit'].max(1)[1].cpu().numpy())):
                    qid2ans[qid] = dataset.label2ans[l]
                    result_score[qid] = target[idx, l].item()
                
                tbar.update(1)
            tbar.close()
        all_acc = evaluator.evaluate(qid2ans)
        print(f"All accuracy on {self.args.cs_test}: {100 * all_acc:.2f}")
        return result_score
    
    def load_data(self):
        from torch.utils.data.dataloader import DataLoader
        if self.args.cs_test == 'vqa_rep':
            from data_loader.vqar_dataset import VQARephraseDataset
            dataset = VQARephraseDataset(self.args)
            evaluator = VQAEvaluator(dataset)
            data_loader = DataLoader(
                dataset,
                batch_size=1024,
                shuffle=False,
                drop_last=False,
                num_workers=8,
                pin_memory=True
            )
            return DataTuple(dataset=dataset, loader=data_loader,
                             evaluator=evaluator)
        elif self.args.cs_test == 'vqa_p2':
            from data_loader.vqap2_dataset import VQAP2Dataset
            dataset = VQAP2Dataset(self.args)
            evaluator = VQAEvaluator(dataset)
            data_loader = DataLoader(
                dataset,
                batch_size=1024,
                shuffle=False,
                drop_last=False,
                num_workers=8,
                pin_memory=True
            )
            return DataTuple(dataset=dataset, loader=data_loader,
                             evaluator=evaluator)
        elif self.args.cs_test == 'vqa_ce':
            from data_loader.vqace_dataset import VQACEDataset
            dataset = VQACEDataset(self.args, test_mode=self.test_mode)
            evaluator = VQAEvaluator(dataset)
            data_loader = DataLoader(
                dataset,
                batch_size=1024,
                shuffle=False,
                drop_last=False,
                num_workers=8,
                pin_memory=True
            )
            return DataTuple(dataset=dataset, loader=data_loader,
                             evaluator=evaluator)
        elif self.args.cs_test == 'iv_vqa':
            from data_loader.ivvqa_dataset import IVVQADataset
            dataset = IVVQADataset(self.args, test_mode=self.test_mode)
            evaluator = VQAEvaluator(dataset)
            data_loader = DataLoader(
                dataset,
                batch_size=1024,
                shuffle=False,
                drop_last=False,
                num_workers=8,
                pin_memory=True
            )
            return DataTuple(dataset=dataset, loader=data_loader,
                             evaluator=evaluator)
        elif self.args.cs_test == 'cv_vqa':
            from data_loader.cvvqa_dataset import CVVQADataset
            dataset = CVVQADataset(self.args, test_mode=self.test_mode)
            evaluator = VQAEvaluator(dataset)
            data_loader = DataLoader(
                dataset,
                batch_size=1024,
                shuffle=False,
                drop_last=False,
                num_workers=8,
                pin_memory=True
            )
            return DataTuple(dataset=dataset, loader=data_loader,
                             evaluator=evaluator)
        else:
            raise ImportError("evaluation dataset does not exist.")
    
    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)


if __name__ == '__main__':
    from param import args
    
    # testing vanilla LXMERT
    from vqa_model import VQAModel
    # LXMERT + CIB
    # from baseline.lxmert.vqa_model import CIBVQAModelV2 as VQAModel
    
    # args.load = 'snap/baseline/lxmert/BEST'
    for cs_test in ['vqa_rep', 'vqa_p2', 'vqa_ce', 'iv_vqa', 'cv_vqa']:
        print(f"{'*' * 80}")
        print(f"Evaluating on {cs_test}.")
        args.cs_test = cs_test
        cs_evaluation = ConsistencyEvaluation(args, model=VQAModel)

