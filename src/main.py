import argparse
import logging
import os
import pickle
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from model import create_model_FM, Att_FM_model
from trainer import model_train
from utils import Data_Train, Data_Val, Data_Test

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='amazon_beauty', help='Dataset name: amazon_beauty, steam, ml-100k, yelp')
parser.add_argument('--log_file', default='log/', help='log dir path')
parser.add_argument('--random_seed', type=int, default=1997, help='Random seed')  
parser.add_argument('--max_len', type=int, default=50, help='The max length of sequence')
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
parser.add_argument('--num_gpu', type=int, default=1, help='Number of GPU')
parser.add_argument('--batch_size', type=int, default=512, help='Batch Size')  
parser.add_argument("--hidden_size", default=128, type=int, help="hidden size of model")
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout of representation')
parser.add_argument('--emb_dropout', type=float, default=0.3, help='Dropout of item embedding')
parser.add_argument("--hidden_act", default="gelu", type=str) # gelu relu
parser.add_argument('--num_blocks', type=int, default=4, help='Number of Transformer blocks')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs for training')  ## 500
parser.add_argument('--decay_step', type=int, default=100, help='Decay step for StepLR')
parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for StepLR')
parser.add_argument('--metric_ks', nargs='+', type=int, default=[5, 10, 20], help='ks for Metric@k')
parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'])
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='L2 regularization')
parser.add_argument('--momentum', type=float, default=None, help='SGD momentum')
parser.add_argument('--lambda_uncertainty', type=float, default=0.001, help='uncertainty weight')
parser.add_argument('--eval_interval', type=int, default=5, help='the number of epoch to eval')
parser.add_argument('--patience', type=int, default=5, help='the number of epoch to wait before early stop')
parser.add_argument('--eps', type=float, default=0.001, help='start step')
parser.add_argument('--sample_N', type=int, default=30, help='Euler calculate steps')
parser.add_argument('--lambda_t', type=float, default=0.001, help='scale of condition')
parser.add_argument('--dropout_c', type=float, default=0.1, help='scale of condition')
parser.add_argument('--eps_reverse', type=float, default=0.001, help='reverse start step')
parser.add_argument('--m_logNorm', type=float, default=1.0, help='Logit-Normal Sampling mean')
parser.add_argument('--s_logNorm', type=float, default=0.6, help='Logit-Normal Sampling Variance')
parser.add_argument('--s_modsamp', type=float, default=1.0, help='Mode_sample_timestep scale parameter')
parser.add_argument('--last', type=int, default=2, help='last H Get')
parser.add_argument('--mask_ratio', type=float, default=1.0, help='Balanced positive-negative class ratio')
parser.add_argument('--sampling_method', type=str, default='mode', choices=['uniform', 'logit_normal', 'mode', 'cosmap'])
parser.add_argument('--Loss_Alpha', type=float, default=0.2, help='Loss parameter')
parser.add_argument('--Loss_Beta', type=float, default=0.4, help='Loss parameter')
args = parser.parse_args()

print(args)

if not os.path.exists(args.log_file):
    os.makedirs(args.log_file)
if not os.path.exists(args.log_file + args.dataset):
    os.makedirs(args.log_file + args.dataset )


logging.basicConfig(level=logging.INFO, filename=args.log_file + args.dataset + '/' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '.log',
                    datefmt='%Y/%m/%d %H:%M:%S', format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s', filemode='w')
logger = logging.getLogger(__name__)
logger.info(args)


def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def item_num_create(args, item_num):
    args.item_num = item_num + 1
    return args

def user_num_create(args, user_num):
    args.user_num = user_num + 1
    return args

def main(args,logger):    
    fix_random_seed_as(args.random_seed)
    path_data = '../datasets/data/' + args.dataset + '/dataset.pkl'
    with open(path_data, 'rb') as f:
        data_raw = pickle.load(f)
    
    args = item_num_create(args, len(data_raw['smap']))
    args = user_num_create(args, len(data_raw['train'].items()))

    tra_data = Data_Train(data_raw['train'], args)
    val_data = Data_Val(data_raw['train'], data_raw['val'], args)
    test_data = Data_Test(data_raw['train'], data_raw['val'], data_raw['test'], args)
    tra_data_loader = tra_data.get_pytorch_dataloaders()
    val_data_loader = val_data.get_pytorch_dataloaders()
    test_data_loader = test_data.get_pytorch_dataloaders()
    FM_rec = create_model_FM(args)
    rec_fm_joint_model = Att_FM_model(FM_rec, args)
    
    best_model, test_results = model_train(tra_data_loader, val_data_loader, test_data_loader, rec_fm_joint_model, args, logger)


if __name__ == '__main__':
    main(args,logger)