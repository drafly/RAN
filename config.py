import argparse

def getConfig():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, default='train', help='Model Training or Testing options')
    parser.add_argument('--exp_num', default=7, type=str, help='experiment_number')
    parser.add_argument('--dataset', type=str, default='Nature', help='dataset name')
    parser.add_argument('--data_path', type=str, default='/data/dataset/wangyi/TRACER')

    # Training parameter settings
    parser.add_argument('--img_size', type=int, default=384)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lr_factor', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=5, help="Scheduler ReduceLROnPlateau's parameter & Early Stopping(+5)")
    parser.add_argument('--model_path', type=str, default='/data/models/wangyi/EfficientNet/results/')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_map', type=bool, default=None, help='Save prediction map')
    parser.add_argument('--result_path', type=str, default='/data/dataset/wangyi/EfficientNet/results/')

    cfg = parser.parse_args()

    return cfg


if __name__ == '__main__':
    cfg = getConfig()
    cfg = vars(cfg)
    print(cfg)