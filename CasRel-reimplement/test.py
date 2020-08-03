import config
import framework
import argparse
import models
import os
import torch
import numpy as np
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='Casrel', help='name of the model')
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--multi_gpu', type=bool, default=False)
parser.add_argument('--dataset', type=str, default='CMED')
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--max_epoch', type=int, default=200)
parser.add_argument('--test_epoch', type=int, default=1)
parser.add_argument('--train_prefix', type=str, default='train_triples')
parser.add_argument('--dev_prefix', type=str, default='dev_triples')
parser.add_argument('--test_prefix', type=str, default='test_triples')
parser.add_argument('--max_len', type=int, default=150)
parser.add_argument('--rel_num', type=int, default=44)
parser.add_argument('--period', type=int, default=50)
parser.add_argument('--debug', type=bool, default=False)
args = parser.parse_args()

con = config.Config(args)

fw = framework.Framework(con)

model = {
    'Casrel': models.Casrel
}

fw.testall(model[args.model_name])
