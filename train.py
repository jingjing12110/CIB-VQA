# @File :train.py
# @Time :2021/7/12
# @Desc :
import os
import json
import collections
from tqdm import tqdm
from datetime import datetime
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup, \
    get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup

from data_loader.base_dataset import DataLoaderX as DataLoader
from vqa_data import VQAv2Dataset, AdVQADataset, AVQADataset
from data_loader.base_dataset import VQAEvaluator
from lxrt.tokenization import BertTokenizer
from lxrt.entry import convert_sents_to_features
from lxrt.optimization import BertAdam
from vqa_model import CIBVQAModelV2 as VQAModel

from utils import save_json
from param import args

TIMESTAMP = "{0:%Y-%m-%dT-%H-%M-%S/}".format(datetime.now())
DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


def get_data_tuple(splits: str, bs: int, num_workers,
                   shuffle=False, drop_last=False):
    dset = VQAv2Dataset(splits)
    evaluator = VQAEvaluator(dset)
    data_loader = DataLoader(
        dset,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True
    )
    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


def get_data_tuple_adv(dataset, test_mode: str, bs: int, num_workers,
                       shuffle=False, drop_last=False):
    dset = dataset(test_mode)
    evaluator = VQAEvaluator(dset)
    data_loader = DataLoader(
        dset,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True
    )
    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class VQA:
    def __init__(self, args):
        self.args = args
        # Dataset
        self.train_tuple = get_data_tuple(
            args.train,
            bs=args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            drop_last=True
        )
        if args.valid != "":
            self.valid_tuple = get_data_tuple(
                args.valid,
                bs=512,
                num_workers=self.args.num_workers,
                shuffle=False,
                drop_last=False
            )
        else:
            self.valid_tuple = None
        # Using the bert tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )
        self.max_seq_length = 20
        # Model
        self.model = VQAModel(
            self.train_tuple.dataset.num_answers,
            using_cib=self.args.using_cib,
            mi_lb=self.args.mi_lb)
        
        # Load LXMERT pre-trained weights
        self.model.load("snap/pretrained/model")
        self.model = self.model.cuda()
        
        # ***************************************************************
        self.bce_loss = nn.BCEWithLogitsLoss()  # task_loss
        if self.args.using_cib:
            self.beta = self.args.beta
            self.alpha = self.args.alpha
            self.mi_mode = 'sample'
        # ***************************************************************
        
        #  Optimizer
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("BertAdam Total Iters: %d" % t_total)
            
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=args.warmup_ratio,
                                  e=8e-6,
                                  t_total=t_total)
        else:
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler(
                lr_mode=self.args.lr_mode
            )
        
        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)
        self.writer = None
        if args.test is None and args.tf_writer:
            self.writer = SummaryWriter(os.path.join(
                self.output, f'{TIMESTAMP}/logs'))
    
    def train_org(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))
                        ) if self.args.tqdm else (lambda x: x)
        
        best_valid, total_loss, train_iter = 0., 0., 0
        self.model.zero_grad()
        
        for epoch in range(self.args.epochs):
            quesid2ans = {}
            for i, (ques_id, feat, pos, sent, target) in iter_wrapper(
                    enumerate(loader)):
                # data preparing
                feat, pos, target = feat.cuda(), pos.cuda(), target.cuda()
                
                train_features = convert_sents_to_features(
                    sent, self.max_seq_length, self.tokenizer)
                input_ids = torch.tensor([f.input_ids for f in train_features],
                                         dtype=torch.long).cuda()
                input_mask = torch.tensor([f.input_mask for f in train_features],
                                          dtype=torch.long).cuda()
                segment_ids = torch.tensor(
                    [f.segment_ids for f in train_features],
                    dtype=torch.long).cuda()
                
                outputs = self.model(
                    input_ids=input_ids,
                    segment_ids=segment_ids,
                    input_mask=input_mask,
                    feat=feat, pos=pos,
                )
                # bce loss
                loss = self.bce_loss(
                    outputs['logit'], target) * target.size(1)
                loss.backward()
                
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()
                if self.args.optim != 'bert':
                    self.lr_scheduler.step()
                self.optim.zero_grad()
                self.model.zero_grad()
                
                total_loss += loss
                for qid, l in zip(
                        ques_id, outputs['logit'].max(1)[1].cpu().numpy()):
                    quesid2ans[qid.item()] = dset.label2ans[l]  # ans
                
                if self.args.tf_writer:
                    self.writer.add_scalar(
                        'Train/batch_loss',
                        loss,
                        train_iter
                    )
                    self.writer.add_scalar(
                        'Train/average_loss',
                        total_loss / (train_iter + 1),
                        train_iter
                    )
                train_iter += 1
            
            # *compute train score*
            train_score = evaluator.evaluate(quesid2ans) * 100.
            log_str = f"\nEpoch {epoch}: Train {train_score:.2f}\n" \
                      f"Epoch {epoch}: loss {total_loss / (train_iter + 1):.2f}"
            if self.args.tf_writer:
                self.writer.add_scalar(
                    'Train/acc',
                    train_score,
                    epoch
                )
            # *Do Validation*
            if self.valid_tuple is not None:
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")
                # self.save(f"model_{epoch}")
                if self.args.tf_writer:
                    self.writer.add_scalar(
                        'Val/acc',
                        valid_score,
                        epoch
                    )
                log_str += f"\nEpoch {epoch}: " \
                           f"Val {valid_score * 100.:.2f}\n" + \
                           f"Epoch {epoch}: " \
                           f"Best {best_valid * 100.:.2f}\n"
            print(log_str, end='')
            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()
            if self.args.test_on_advqa:
                self.evaluate_on_advqa("val")
            if self.args.test_on_avqa:
                self.evaluate_on_avqa("val")
        # self.save("LAST")
        if self.args.tf_writer:
            self.writer.close()
    
    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        best_valid, total_loss, train_iter = 0., 0., 0
        
        self.model.zero_grad()
        self.optim.zero_grad()
        for epoch in range(self.args.epochs):
            self.model.train()
            task_loss, cib_loss, mi_lg_loss = 0., 0., 0.
            quesid2ans = {}
            pbar = tqdm(total=len(loader), ncols=120)
            for i, (ques_id, feat, pos, sent, target) in enumerate(loader):
                # data preparing
                feat, pos, target = feat.cuda(), pos.cuda(), target.cuda()
                
                train_features = convert_sents_to_features(
                    sent, self.max_seq_length, self.tokenizer)
                input_ids = torch.tensor([f.input_ids for f in train_features],
                                         dtype=torch.long).cuda()
                input_mask = torch.tensor([f.input_mask for f in train_features],
                                          dtype=torch.long).cuda()
                segment_ids = torch.tensor(
                    [f.segment_ids for f in train_features],
                    dtype=torch.long).cuda()
                
                outputs = self.model(
                    input_ids=input_ids,
                    segment_ids=segment_ids,
                    input_mask=input_mask,
                    feat=feat, pos=pos, )
                # task loss
                loss = self.bce_loss(
                    outputs['logit'], target) * target.size(1)
                task_loss += loss.item()
                
                # CIB constraint
                cib_loss_i = outputs['cib'] * self.beta
                cib_loss += cib_loss_i
                
                # MI between local-global representation
                mi_lg_loss_i = outputs['mi_lg'] * self.alpha
                mi_lg_loss += mi_lg_loss_i
                
                loss += cib_loss_i + mi_lg_loss_i
                
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()
                if self.args.optim != 'bert':
                    self.lr_scheduler.step()
                self.optim.zero_grad()
                self.model.zero_grad()
                
                total_loss += loss.item()
                for qid, l in zip(
                        ques_id, outputs['logit'].max(1)[1].cpu().numpy()):
                    quesid2ans[qid.item()] = dset.label2ans[l]  # ans
                
                if self.args.tf_writer:
                    self.writer.add_scalar(
                        'Train/batch_loss',
                        loss,
                        train_iter
                    )
                    # self.writer.add_scalar(
                    #     'Train/cib_loss',
                    #     cib_loss_i,
                    #     train_iter
                    # )
                    # self.writer.add_scalar(
                    #     'Train/upper_bound_l',
                    #     outputs["upper_bound_l"],
                    #     train_iter
                    # )
                    # self.writer.add_scalar(
                    #     'Train/upper_bound_v',
                    #     outputs["upper_bound_v"],
                    #     train_iter
                    # )
                    # self.writer.add_scalar(
                    #     'Train/lower_bound_lv',
                    #     outputs["lower_bound_lv"],
                    #     train_iter
                    # )
                    # self.writer.add_scalar(
                    #     'Train/skl',
                    #     outputs["skl"],
                    #     train_iter
                    # )
                    # self.writer.add_scalar(
                    #     'Train/mi_lg',
                    #     mi_lg_loss_i,
                    #     train_iter
                    # )
                    # self.writer.add_scalar(
                    #     'Train/average_loss',
                    #     total_loss / (train_iter + 1),
                    #     train_iter
                    # )
                    # self.writer.add_scalar(
                    #     'Train/task_loss',
                    #     task_loss / (train_iter + 1),
                    #     train_iter
                    # )
                train_iter += 1
                desc_str = f'Epoch {epoch} | loss {loss:.4f} ' \
                           f'| cib {cib_loss_i:.4f} | mi_lg {mi_lg_loss_i:.4f}'
                pbar.set_description(desc_str)
                pbar.update(1)
            pbar.close()
            # *compute train score*
            train_score = evaluator.evaluate(quesid2ans) * 100.
            log_str = \
                f"\nEpoch {epoch}: Train {train_score:.2f}\n" \
                f"Epoch {epoch}: task_loss {task_loss / (i + 1):.2f}\n" \
                f"Epoch {epoch}: cib_loss {cib_loss / (i + 1):.3f}\n" \
                f"Epoch {epoch}: mi_lv {mi_lg_loss / (i + 1):.3f}\n" \
                f"Epoch {epoch}: total_loss {total_loss / (train_iter + 1):.2f}"
            if self.args.tf_writer:
                self.writer.add_scalar(
                    'Train/acc',
                    train_score,
                    epoch
                )
            # *Do Validation*
            if self.valid_tuple is not None:
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")
                # self.save(f"model_{epoch}")
                if self.args.tf_writer:
                    self.writer.add_scalar(
                        'Val/acc',
                        valid_score,
                        epoch
                    )
                log_str += f"\nEpoch {epoch}: " \
                           f"Val {valid_score * 100.:.2f}\n" + \
                           f"Epoch {epoch}: " \
                           f"Best {best_valid * 100.:.2f}\n"
            print(log_str, end='')
            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()
            # test on advqa val
            if self.args.test_on_advqa:
                self.evaluate_on_advqa("val", epoch=f"{epoch}", is_save=True)
                self.evaluate_on_advqa("test", epoch=f"{epoch}")
            if self.args.test_on_avqa:
                self.evaluate_on_avqa("val", epoch=f"{epoch}", is_save=True)
                self.evaluate_on_avqa("test", epoch=f"{epoch}")
        # self.save("LAST")
        if self.args.tf_writer:
            self.writer.close()
    
    def vl_training_step(
            self, input_ids, input_mask, segment_ids, feat, pos, target):
        self.model.train()
        
        task_loss, upperbound_loss, lowerbound_loss = 0.0, 0.0, 0.0
        outputs = self.model(
            input_ids=input_ids,
            segment_ids=segment_ids,
            input_mask=input_mask,
            feat=feat, pos=pos,
        )
        # bce loss
        loss = self.bce_loss(
            outputs['logit'], target.cuda()
        ) * target.size(1)
        task_loss += loss.item()
        
        # CIB
        input_len = torch.sum(input_mask, 1)  # [bs]
        embeddings = [outputs['local_linguistic_embedding'][i, :l]
                      for i, l in enumerate(input_len)]
        upper_bound_l = self.model.mi_upper_estimator.update(
            y_samples=torch.cat(embeddings, dim=0),  # [-1, 768]
            mi_mode=self.mi_mode
        )
        upper_bound_v = self.model.mi_upper_estimator.update(
            y_samples=outputs['local_visual_embedding'].view(-1, 768),
            mi_mode=self.mi_mode
        )
        additional_term = self.model.joint_mi_estimator(
            outputs['local_linguistic_embedding'],
            outputs['local_visual_embedding']
        )
        upper_bound = (upper_bound_l + upper_bound_v
                       ) * self.beta + additional_term
        loss += upper_bound * target.size(1)
        loss.backward()
        
        upperbound_loss += upper_bound.item()
        return {
            "task_loss": task_loss,
            "upperbound_loss": upperbound_loss,
            "lowerbound_loss": lowerbound_loss,
            "logit": outputs['logit'],
            "additional_term": additional_term,
            "total_loss": loss.detach(),
        }

    def evaluate_on_avqa(self, test_mode, epoch='BEST', is_save=False):
        data_tuple = get_data_tuple_adv(
            AVQADataset,
            test_mode=test_mode,
            bs=512,
            num_workers=self.args.num_workers,
            shuffle=False,
            drop_last=False)
        qid2ans = self.predict(data_tuple)
        if test_mode == "val":
            avqa_score = data_tuple.evaluator.evaluate(qid2ans)
            m_txt = open(os.path.join(args.output, f'log.log'), 'a+')
            m_txt.write(f'Epoch {epoch}: avqa_val_acc: {100 * avqa_score:.2f}\n')
            m_txt.close()
            print(f'Epoch {epoch}: avqa val acc: {100 * avqa_score:.2f}')
        
            if is_save:  # format convert
                with open(f"{self.output}/avqa_val_epoch{epoch}.jsonl", 'w'
                          ) as outfile:
                    for qid, ans in qid2ans.items():
                        entry = {"uid": qid, "answer": f"{ans}"}
                        json.dump(entry, outfile)
                        outfile.write('\n')
        elif test_mode == "test":
            with open(f"{self.output}/avqa_test_epoch{epoch}.jsonl", 'w'
                      ) as outfile:
                for qid, ans in qid2ans.items():
                    entry = {"uid": qid, "answer": f"{ans}"}
                    json.dump(entry, outfile)
                    outfile.write('\n')

    def evaluate_on_advqa(self, test_mode, epoch='BEST', is_save=False):
        data_tuple = get_data_tuple_adv(
            AdVQADataset,
            test_mode=test_mode,
            bs=512,
            num_workers=self.args.num_workers,
            shuffle=False,
            drop_last=False)
        qid2ans = self.predict(data_tuple)
        if test_mode == "val":
            advqa_score = data_tuple.evaluator.evaluate(qid2ans)
            m_txt = open(os.path.join(args.output, f'log.log'), 'a+')
            m_txt.write(
                f'Epoch {epoch}: advqa_val_acc: {100 * advqa_score:.2f}\n')
            m_txt.close()
            print(f'Epoch {epoch}: advqa val acc: {100 * advqa_score:.2f}')
            
            if is_save:  # format convert
                with open(f"{self.output}/advqa_val_epoch{epoch}.jsonl", 'w'
                          ) as outfile:
                    for qid, ans in qid2ans.items():
                        entry = {"uid": qid, "answer": f"{ans}"}
                        json.dump(entry, outfile)
                        outfile.write('\n')
        elif test_mode == "test":
            with open(f"{self.output}/advqa_test_epoch{epoch}.jsonl", 'w'
                      ) as outfile:
                for qid, ans in qid2ans.items():
                    entry = {"uid": qid, "answer": f"{ans}"}
                    json.dump(entry, outfile)
                    outfile.write('\n')
    
    def predict(self, eval_tuple: DataTuple, dump=None):
        """Predict the answers to questions in a data split.
        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        tbar = tqdm(total=len(loader), ascii=True, desc='Val', ncols=80)
        for i, datum_tuple in enumerate(loader):
            # Avoid seeing ground truth
            ques_id, feat, pos, sent = datum_tuple[:4]
            with torch.no_grad():
                # output = self.model(feats.cuda(), boxes.cuda(), sent)
                train_features = convert_sents_to_features(
                    sent, self.max_seq_length, self.tokenizer)
                input_ids = torch.tensor([f.input_ids for f in train_features],
                                         dtype=torch.long).cuda()
                input_mask = torch.tensor([f.input_mask for f in train_features],
                                          dtype=torch.long).cuda()
                segment_ids = torch.tensor(
                    [f.segment_ids for f in train_features],
                    dtype=torch.long).cuda()
                
                outputs = self.model(
                    input_ids=input_ids,
                    segment_ids=segment_ids,
                    input_mask=input_mask,
                    feat=feat.cuda(), pos=pos.cuda(),
                )
                for qid, l in zip(
                        ques_id, outputs['logit'].max(1)[1].cpu().numpy()):
                    quesid2ans[qid.item()] = dset.label2ans[l]
            tbar.update(1)
        tbar.close()
        self.model.train()
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans
    
    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        # quesid2ans = self.predict(eval_tuple, dump)
        return eval_tuple.evaluator.evaluate(self.predict(eval_tuple, dump))
    
    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))
    
    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)
    
    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, target) in enumerate(loader):
            _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                quesid2ans[qid.item()] = ans
        return evaluator.evaluate(quesid2ans)
    
    def create_optimizer_and_scheduler(self, lr_mode='linear'):
        batch_per_epoch = len(self.train_tuple.loader)
        t_total = int(batch_per_epoch * self.args.epochs)
        warmup_iters = int(t_total * self.args.warmup_ratio)
        # warmup_iters = 1000
        
        # if self.verbose:
        print("Batch per epoch: %d" % batch_per_epoch)
        print("Total Iters: %d" % t_total)
        print('Warmup ratio:', self.args.warmup_ratio)
        print("Warm up Iters: %d" % warmup_iters)
        
        no_decay = ["bias", "LayerNorm.weight"]
        if self.args.using_cib:
            mi_lb_params = list(map(id, self.model.mi_lb_estimator.parameters()))
            mi_ub_params = list(map(id, self.model.mi_ub_estimator.parameters()))
            base_params = filter(
                lambda p: id(p) not in mi_lb_params + mi_ub_params,
                self.model.parameters())
            optimizer_grouped_parameters = [
                {"params": base_params},
                {"params": self.model.mi_lb_estimator.parameters(),
                 'lr': self.args.lr * 100},
                {"params": self.model.mi_ub_estimator.parameters(),
                 'lr': self.args.lr * 100},
            ]
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if
                               not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if
                               any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
        optim = AdamW(optimizer_grouped_parameters, self.args.lr)
        lr_scheduler = None
        if lr_mode == 'linear':  # linear
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=optim,
                num_warmup_steps=warmup_iters,
                num_training_steps=t_total
            )
        elif lr_mode == 'cosine':  # cosine
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=optim,
                num_warmup_steps=warmup_iters,
                num_training_steps=t_total,
                num_cycles=0.8,
            )
        elif lr_mode == 'cosine_hard':
            lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer=optim,
                num_warmup_steps=warmup_iters,
                num_training_steps=t_total,
                num_cycles=4.0,
            )
        elif lr_mode == 'constant':
            lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=optim,
                num_warmup_steps=warmup_iters,
            )
        return optim, lr_scheduler


if __name__ == "__main__":
    print(args)
    # Build Class
    vqa = VQA(args)
    
    # Load VQA model weights
    if args.load is not None:
        vqa.load(args.load)
    
    # Testing
    if args.test is not None:
        # test on VQA v2
        if 'test' in args.test:
            vqa.predict(
                get_data_tuple(
                    args.test,
                    bs=args.batch_size,
                    num_workers=args.num_workers,
                    shuffle=False,
                    drop_last=False),
                dump=os.path.join(
                    args.output, f'test_predict_{args.test_name}.json')
            )
        elif 'minival' in args.test:
            acc = vqa.evaluate(
                get_data_tuple('minival',
                               bs=args.batch_size,
                               num_workers=args.num_workers,
                               shuffle=False,
                               drop_last=False),
                # dump=os.path.join(args.output, 'minival_predict.json')
            )
            m_txt = open(os.path.join(args.output, f'log.log'), 'a+')
            m_txt.write(
                f'\n minival_acc: {100 * acc:.2f}\n')
            m_txt.close()
            print(f'minival acc: {100 * acc:.2f}')
        else:
            assert False, "No such test option for %s" % args.test
        # testing on advqa
        if args.test_on_advqa:
            vqa.evaluate_on_advqa("val", is_save=True)
            vqa.evaluate_on_advqa("test")
        if args.test_on_avqa:
            vqa.evaluate_on_avqa("val", is_save=True)
            vqa.evaluate_on_avqa("test")
    else:
        print('Splits in Train data:', vqa.train_tuple.dataset.splits)
        if vqa.valid_tuple is not None:
            save_json(vars(args), os.path.join(args.output, f'args.json'))
            print('Splits in Valid data:', vqa.valid_tuple.dataset.splits)
            # print("Valid Oracle: %0.2f" % (
            #         vqa.oracle_score(vqa.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        if args.using_cib:
            vqa.train(vqa.train_tuple, vqa.valid_tuple)
        else:
            vqa.train_org(vqa.train_tuple, vqa.valid_tuple)
