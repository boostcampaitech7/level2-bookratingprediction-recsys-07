import argparse
import torch
import random
import os
import numpy as np

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    args.seed = 42
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.predict = False
    args.train = {'epochs': 10, 'save_best_model': True, 'resume': False, 'ckpt_dir': 'checkpoints/'}
    args.dataloader = {'batch_size': 64, 'shuffle': True, 'num_workers': 4}
    args.dataset = {'data_path': '/path/to/data/', 'valid_ratio': 0.2}
    args.optimizer = {'type': 'Adam', 'args': {'lr': 0.001, 'weight_decay': 1e-4}}
    args.loss = 'RMSELoss'
    args.metrics = ['RMSELoss', 'MSELoss']
    args.model = 'DCN'
    args.model_args = {
        'DCN': {
            'datatype': 'context',
            'embed_dim': 16,
            'cross_layer_num': 3,
            'mlp_dims': [16, 32],
            'batchnorm': True,
            'dropout': 0.2
        }
    }
    seed_everything(args.seed)
    
    '''
    1. train,valid 나누기
    2. 전체 train에 대해서
    3. sub 파일 만들가
    4. 최적화
    '''



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DCN')
    arg = parser.add_argument
    