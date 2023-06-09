import os
import pprint
import random
import warnings
import torch
import numpy as np
from trainer import Trainer, Tester

from config import getConfig
warnings.filterwarnings('ignore')
args = getConfig()


def main(args):

    # Random Seed
    seed = args.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.action == 'train':
        print('<---- Training Params ---->')
        pprint.pprint(args)
        save_path = os.path.join(args.model_path, args.dataset, f'experience_{str(args.exp_num)}')

        # Create model directory
        os.makedirs(save_path, exist_ok=True)
        Trainer(args, save_path)

    elif args.action == 'test':
        save_path = os.path.join(args.model_path, args.dataset, f'experience_{str(args.exp_num)}')
        datasets = ['DUT', 'CUHK', 'CTCUG']

        for dataset in datasets:
            args.dataset = dataset
            test_loss, test_mae, test_maxf, test_avgf, test_s_m = Tester(args, save_path).test()

            print(f'Test Loss:{test_loss:.3f} | MAX_F:{test_maxf:.4f} '
                  f'| AVG_F:{test_avgf:.4f} | MAE:{test_mae:.4f} | S_Measure:{test_s_m:.4f}')
    else:
        print("Please choose the action 'train' or 'test'. ")


if __name__ == '__main__':
    main(args)