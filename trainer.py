# @File :trainer.py
# @Time :2021/4/28
# @Desc :
import os
from tqdm import tqdm
from datetime import datetime
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.functional as F
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from pretrain.qa_answer_table import load_lxmert_qa
from data_loader.vqav2_dataset import VQAv2LTransDataset as VQAv2Dataset
from data_loader.base_dataset import VQAEvaluator, DataTuple
from vqa_model import VQAModel


class Trainer:
    def __init__(self, args, save_epoch_weight=False, verbose=True):
        self.args = args
        self.save_epoch_weight = save_epoch_weight
        self.verbose = verbose
        
        # Model
        self.model = VQAModel(self.args)
        # Load pre-trained weights
        # if args.load_lxmert is not None:
        #     self.model.lxrt_encoder.load(args.load_lxmert)
        # if args.load_lxmert_qa is not None:
        #     load_lxmert_qa(args.load_lxmert_qa, self.model,
        #                    label2ans=self.train_tuple.dataset.label2ans)
        
        # GPU options
        self.model = self.model.cuda()
        
        # Load current model weights
        self.start_epoch = 0
        if args.load is not None:
            self.load(self.args.load)
        
        if args.multiGPU:
            self.model.lvc_encoder.multi_gpu()
        
        self.train_tuple = self.get_data_tuple(
            args.train,
            bs=args.batch_size,
            shuffle=True,
            drop_last=True
        )
        
        if self.args.valid:
            self.valid_tuple = self.get_data_tuple(
                args.valid,
                bs=args.batch_size,
                shuffle=False,
                drop_last=False
            )
        else:
            self.valid_tuple = None
        
        # Loss and Optimizer
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='sum')
        # information bottleneck loss
        self.beta = 1e-3
        
        if 'bert' in args.optim:
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()
        else:
            self.optim = args.optimizer(
                self.model.parameters(),
                self.args.lr)
        
        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)
        self.writer = None
        if self.args.test is None and args.tf_writer:
            time_stamp = "{0:%Y-%m-%dT-%H-%M-%S/}".format(datetime.now())
            self.writer = SummaryWriter(os.path.join(
                self.output, f'{time_stamp}/logs'))
        self.train_iter = 0
    
    def train(self):
        self.model.train()
        dataset, loader, evaluator = self.train_tuple
        best_valid, total_loss = 0., 0.
        
        for epoch in range(self.start_epoch, self.args.epochs):
            qid2ans = {}
            tbar = tqdm(total=len(loader), ncols=80, desc='Train')
            for i, data_tuple in enumerate(loader):
                self.optim.zero_grad()

                qids, visual_feats, visual_pos, sent, target, input_ids, \
                    input_mask, segment_ids, graph_arc, graph_obj = data_tuple
                
                visual_feats, visual_pos = visual_feats.cuda(), visual_pos.cuda()
                
                input_ids, input_mask = input_ids.cuda(), input_mask.cuda()
                segment_ids, graph_arc = segment_ids.cuda(), graph_arc.cuda()
                graph_obj = graph_obj.cuda()

                output = self.model(
                    visual_feats=visual_feats,
                    visual_pos=visual_pos,
                    sent=sent,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    graph_arc=graph_arc,
                    graph_obj=graph_obj,
                    is_training=True,
                )
                logit = output['logit']
                
                # loss = self.bce_loss(logit, target.cuda()) * logit.size(1)
                bce = self.bce_loss(logit, target.cuda())
                kl = 0.5 * torch.sum(output['mu'].pow(2) + output['std'].pow(2)
                                     - 2 * output['std'].log() - 1)
                loss = (self.beta * kl + bce) / logit.size(0)
                
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()
                self.lr_scheduler.step()

                total_loss += loss.detach() / logit.size(0)
                for qid, l in zip(qids, logit.max(1)[1].cpu().numpy()):
                    qid2ans[qid.item()] = dataset.label2ans[l]  # ans
                if self.args.tf_writer:
                    self.writer.add_scalar(
                        'Train/batch_loss',
                        loss.detach(),
                        self.train_iter
                    )
                    # self.writer.add_scalar(
                    #     'Train/average_loss',
                    #     total_loss / (self.train_iter + 1),
                    #     self.train_iter
                    # )
                    # if output['kl']:
                    self.writer.add_scalar(
                        'Train/kl_div',
                        self.beta * kl,
                        self.train_iter
                    )
                    # if output['bce']:
                    self.writer.add_scalar(
                        'Train/bce',
                        bce,
                        self.train_iter
                    )
                    self.train_iter += 1
                tbar.update(1)
            tbar.close()
            
            # *compute train score*
            train_score = evaluator.evaluate(qid2ans) * 100.
            print(f'Epoch {epoch}: train_acc: {train_score:.2f}')
            m_txt = open(os.path.join(self.output, f'log.txt'), 'a+')
            m_txt.write(
                f'Epoch {epoch}:\n'
                f'\t train_acc: {train_score:.2f}\n'
                f'\t train_loss: {total_loss / (self.train_iter + 1):.4f}\n')
            m_txt.close()
            
            if self.args.tf_writer:
                self.writer.add_scalar(
                    'Train/acc',
                    train_score,
                    epoch
                )
            # *Do Validation*
            if self.valid_tuple is not None:
                valid_score = self.evaluate(self.valid_tuple) * 100.
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")
                if self.save_epoch_weight:
                    self.save(f"model_{epoch}")
                if self.args.tf_writer:
                    self.writer.add_scalar(
                        'Val/acc',
                        valid_score,
                        epoch
                    )
                print(f'Epoch {epoch}: val_acc: {valid_score:.2f}\n'
                      f'Epoch {epoch}: best_val_acc: {best_valid:.2f}')
                m_txt = open(os.path.join(self.output, f'log.txt'), 'a+')
                m_txt.write(
                    f'\t val_acc: {valid_score:.2f}\n'
                    f'\t best_val_acc: {best_valid:.2f}\n')
                m_txt.close()
        
        # self.save("LAST")
        if self.args.tf_writer:
            self.writer.close()
    
    def predict(self, eval_tuple: DataTuple, dump=None):
        """Predict the answers to questions in a data split.
        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        qid2ans = {}
        with torch.no_grad():
            dataset, loader, evaluator = eval_tuple
            tbar = tqdm(total=len(loader), ascii=True, ncols=80)
            for i, datum_tuple in enumerate(loader):
                # Avoid seeing ground truth
                qids, visual_feats, visual_pos, sent = datum_tuple[:4]
                input_ids, input_mask, segment_ids, graph_arc, graph_obj = \
                    datum_tuple[-5:]
                
                visual_feats, visual_pos = visual_feats.cuda(), visual_pos.cuda()
                input_ids, input_mask = input_ids.cuda(), input_mask.cuda()
                segment_ids, graph_arc = segment_ids.cuda(), graph_arc.cuda()
                graph_obj = graph_obj.cuda()
                
                output = self.model(
                    visual_feats=visual_feats,
                    visual_pos=visual_pos,
                    sent=sent,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    graph_arc=graph_arc,
                    graph_obj=graph_obj,
                    is_training=False,
                )
                
                for qid, l in zip(qids, output['logit'].max(1)[1].cpu().numpy()):
                    qid2ans[qid.item()] = dataset.label2ans[l]
                tbar.update(1)
            tbar.close()
        if dump is not None:
            evaluator.dump_result(qid2ans, dump)
        return qid2ans
    
    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        qid2ans = self.predict(eval_tuple, dump)
        return eval_tuple.evaluator.evaluate(qid2ans)
    
    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))
    
    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)
    
    def load_pretrained(self):
        if self.args.load_lxmert is not None:
            self.model.lxrt_encoder.load(self.args.load_lxmert)
        
        if self.args.load_lxmert_qa is not None:
            load_lxmert_qa(self.args.load_lxmert_qa,
                           self.model,
                           label2ans=self.train_tuple.dataset.label2ans)

    def get_data_tuple(self, splits, bs, shuffle=False, drop_last=False):
        if self.args.output.split('/')[-1] == 'debug':
            from torch.utils.data.dataloader import DataLoader
        else:
            from data_loader.base_dataset import DataLoaderX as DataLoader
    
        dataset = VQAv2Dataset(splits)
        evaluator = VQAEvaluator(dataset)
        data_loader = DataLoader(
            dataset,
            batch_size=bs,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        return DataTuple(dataset=dataset, loader=data_loader, evaluator=evaluator)

    def create_optimizer_and_scheduler(self):
        batch_per_epoch = len(self.train_tuple.loader)
        t_total = int(batch_per_epoch * self.args.epochs)
        warmup_iters = int(t_total * self.args.warmup_ratio)
    
        if self.verbose:
            print("Batch per epoch: %d" % batch_per_epoch)
            print("Total Iters: %d" % t_total)
            print('Warmup ratio:', self.args.warmup_ratio)
            print("Warm up Iters: %d" % warmup_iters)
    
        no_decay = ["bias", "LayerNorm.weight"]
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
        
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optim,
            num_warmup_steps=warmup_iters,
            num_training_steps=t_total
        )
        return optim, lr_scheduler


def train():
    import os
    from param import args
    from utils import save_json
    
    print(args)
    trainer = Trainer(args)
    
    # Testing
    if args.test is not None:
        args.fast = args.tiny = False  # Always loading all data in test
        if 'test' in args.test:
            trainer.predict(
                trainer.get_data_tuple(
                    args.test,
                    bs=512,
                    shuffle=False,
                    drop_last=False),
                dump=os.path.join(args.output, 'test_predict.json')
            )
        elif 'minival' in args.test:
            acc = trainer.evaluate(
                trainer.get_data_tuple(
                    'minival',
                    bs=512,
                    shuffle=False,
                    drop_last=False)
            )
            m_txt = open(os.path.join(args.output, f'log.txt'), 'a+')
            m_txt.write(
                f'\n minival_acc: {100 * acc:.2f}\n')
            m_txt.close()
            print(f'minival acc: {100 * acc:.2f}')
        else:
            assert False, "No such test option for %s" % args.test
    else:
        print('Splits in Train data:', trainer.train_tuple.dataset.splits)
        if trainer.valid_tuple is not None:
            print('Splits in Valid data:', trainer.valid_tuple.dataset.splits)
            # print("Valid Oracle: %0.2f" % (
            #         vqa.oracle_score(vqa.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        # args.model_args = vars(args.model_args)
        save_json(vars(args), os.path.join(args.output, f'args.json'))
        trainer.train()


if __name__ == '__main__':
    train()

