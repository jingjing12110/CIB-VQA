import ast
import random
import argparse
import numpy as np

import torch


def get_optimizer(optim):
    # Bind the optimizer
    if optim == 'rms':
        print("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        print("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == 'adamw':
        print("Optimizer: Using AdamW")
        optimizer = torch.optim.AdamW
    elif optim == 'adamax':
        print("Optimizer: Using Adamax")
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        print("Optimizer: sgd")
        optimizer = torch.optim.SGD
    elif 'bert' in optim:
        optimizer = 'bert'  # The bert optimizer will be bind later.
    else:
        assert False, "Please add your optimizer %s in the list." % optim
    
    return optimizer


def parse_args():
    parser = argparse.ArgumentParser()

    # Model Config
    # Note: LXRT = L, X, R (three encoders), Transformer
    parser.add_argument("--llayers", default=9, type=int,
                        help='Number of Language layers')
    parser.add_argument("--rlayers", default=5, type=int,
                        help='Number of object Relationship layers.')
    parser.add_argument("--xlayers", default=5, type=int,
                        help='Number of CROSS-modality layers.')

    # Pre-training Config
    parser.add_argument("--taskMatched", dest='task_matched',
                        action='store_const', default=False, const=True)
    parser.add_argument("--taskMaskLM", dest='task_mask_lm', action='store_const',
                        default=False, const=True)
    parser.add_argument("--taskObjPredict", dest='task_obj_predict',
                        action='store_const', default=False, const=True)
    parser.add_argument("--taskQA", dest='task_qa', action='store_const',
                        default=False, const=True)
    parser.add_argument("--visualLosses", dest='visual_losses',
                        default='obj,attr,feat', type=str)
    parser.add_argument("--qaSets", dest='qa_sets', default=None, type=str)
    parser.add_argument("--wordMaskRate", dest='word_mask_rate', default=0.15,
                        type=float)
    parser.add_argument("--objMaskRate", dest='obj_mask_rate', default=0.15,
                        type=float)
    parser.add_argument("--fromScratch", dest='from_scratch',
                        action='store_const', default=False, const=True,
                        help='If none of the --load, --loadLXMERT, --loadLXMERTQA'
                             'is set the model would be trained from scratch. '
                             'If --fromScratch is not specified, the model would '
                             'load BERT-pre-trained weights by default. ')
    # add
    parser.add_argument("--num_rel", default=46, type=int,
                        help='number of types of dependency parsing,'
                             'i.e., the label size in language encoder')
    parser.add_argument("--max_seq_length", default=20, type=int)
    parser.add_argument("--num_answer", default=3129, type=int)
    parser.add_argument("--hid_dim", default=768, type=int)
    
    # Data Splits
    parser.add_argument("--train", default='train')
    parser.add_argument("--valid", default='val')
    parser.add_argument(
        "--test", default=None, help='nominival, test')
    
    # Training Hyper-parameters
    parser.add_argument(
        '--fp16', action='store_const', default=False, const=True)
    parser.add_argument('--bs', dest='batch_size', type=int,
                        default=8)
    parser.add_argument('--optim', default='bert')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--warmup_ratio', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=999, help='random seed')
    parser.add_argument('--tf_writer', default=True, type=ast.literal_eval)
    
    # Debugging
    parser.add_argument('--output', type=str, default='snap/debug')
    parser.add_argument("--fast", action='store_const', default=False, const=True)
    parser.add_argument("--tiny", action='store_const', default=False, const=True)
    parser.add_argument("--tqdm", action='store_const', default=False, const=True)
    
    # Model Loading
    parser.add_argument('--load', type=str, default=None,
                        help='Load the model (usually the fine-tuned model).')
    parser.add_argument('--loadLXMERT', dest='load_lxmert', type=str,
                        default=None,
                        # default='snap/pretrained/Epoch20',
                        help='Load the pre-trained LXMERT model.')
    parser.add_argument('--loadLXMERTQA', dest='load_lxmert_qa', type=str,
                        # default='snap/pretrained/model',
                        default=None,
                        help='Load the pre-trained LXMERT model with QA answer '
                             'head.')
    # Optimization
    parser.add_argument("--mceLoss", dest='mce_loss', action='store_const',
                        default=False, const=True)
    
    # Training configuration
    parser.add_argument(
        "--multiGPU", action='store_const', default=False, const=True)
    parser.add_argument(
        "--numWorkers", dest='num_workers', default=0, type=int)
    parser.add_argument("--lr_mode", default='linear', type=str)
    parser.add_argument("--num_cycle", default=0.5, type=float)
    
    # consistency of input variations
    parser.add_argument('--beta', type=float, default=5e-2)
    parser.add_argument('--alpha', type=float, default=5e-1)
    parser.add_argument('--is_adv_training', default=True, type=ast.literal_eval)
    parser.add_argument(
        "--img_feat_root", default='/home/kaka/Data/vqa2/imgfeat/reserve/')
    parser.add_argument(
        '--cs_test', default='vqa_rep',
        choices=['vqa_rep', 'vqa_p2', 'vqa_ce', 'iv_vqa', 'cv_vqa'],
        help='consistency evaluation data set'
    )
    parser.add_argument("--tmode", default='OOD', type=str,
                        help="['OOD', 'ID']")
    
    parser.add_argument(
        "--mi_upper", type=str, default='CLUBSample',
        help='[CLUBSample, CLUB, ]'
    )
    parser.add_argument(
        "--mi_upper_variational", default=False, type=bool
    )
    parser.add_argument(
        "--mi_lb", type=str, default='InfoNCE',
        help='[InfoNCEv2, MINE, NWJ]'
    )
    parser.add_argument(
        "--using_cib", default=False, type=ast.literal_eval,
    )
    parser.add_argument('--test_name', type=str, default='v1')
    parser.add_argument('--model_scale', type=str, default='base')
    parser.add_argument(
        "--test_on_advqa", default=False, type=ast.literal_eval,
    )
    parser.add_argument(
        "--test_on_avqa", default=False, type=ast.literal_eval,
    )
    # parser.add_argument(
    #     "--baseline", default='lxmert',
    #     choices=['lxmert', 'uniter'])
    
    # Parse the arguments.
    # args = parser.parse_args()
    args, unparsed = parser.parse_known_args()
    
    # Bind optimizer class.
    # args.optimizer = get_optimizer(args.optim)
    
    # Set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    return args


args = parse_args()
