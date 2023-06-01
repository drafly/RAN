import argparse

def getConfig():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, default='train', help='Model Training or Testing options')
    parser.add_argument('--exp_num', default=7, type=str, help='experiment_number')
   # 2是COMB-software-TRUE的TE3 3是TE7的COMB-software-TRUE 4是TE3nature数据集 5是TE7nature数据集 6是没有边缘Nature 7是既没有边缘又没有object attention模块Nature 8是ARM数据集COMB-ARM-TRUE 9数据集为COMB-software-all-bceloss
   # 10数据集为COMB-software-all-APILoss-del_MAE 11为COMB-software-TRUE-bceloss 12为COMB-software-all-APILoss-del_BCE-early_stop(mae) 13为COMB-software-all-APILoss-del_BCE-no-early_stop(patient=100)
   # 14为数据集为COMB-software-all1152-APILoss-del_MAE-early_stop(mae) 15为COMB-ARM-ALL-1152-APILoss-del_MAE-early_stop(mae)
    parser.add_argument('--dataset', type=str, default='Nature', help='DUTS')
    parser.add_argument('--data_path', type=str, default='/data/dataset/wangyi/TRACER')

    # Model parameter settings
    parser.add_argument('--arch', type=str, default='3', help='Backbone Architecture')
    parser.add_argument('--channels', type=list, default=[24, 40, 112, 320])
    parser.add_argument('--RFB_aggregated_channel', type=int, nargs='*', default=[32, 64, 128])
    parser.add_argument('--frequency_radius', type=int, default=16, help='Frequency radius r in FFT')
    parser.add_argument('--denoise', type=float, default=0.93, help='Denoising background ratio')
    parser.add_argument('--gamma', type=float, default=0.1, help='Confidence ratio')

    # Training parameter settings
    parser.add_argument('--img_size', type=int, default=384)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--criterion', type=str, default='API', help='API or bce')
    parser.add_argument('--scheduler', type=str, default='Reduce', help='Reduce or Step')
    parser.add_argument('--aug_ver', type=int, default=2, help='1=Normal, 2=Hard')
    parser.add_argument('--lr_factor', type=float, default=0.1)
    parser.add_argument('--clipping', type=float, default=2, help='Gradient clipping')
    parser.add_argument('--patience', type=int, default=5, help="Scheduler ReduceLROnPlateau's parameter & Early Stopping(+5)")
    parser.add_argument('--model_path', type=str, default='/data/models/wangyi/EfficientNet/results/')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_map', type=bool, default=None, help='Save prediction map')


    # Hardware settings
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=4)
    cfg = parser.parse_args()

    return cfg


if __name__ == '__main__':
    cfg = getConfig()
    cfg = vars(cfg)
    print(cfg)