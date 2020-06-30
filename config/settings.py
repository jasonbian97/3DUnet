import argparse


def parse_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--n_channels',
        default=1,
        type=int,
        help=""
    )
    parser.add_argument(
        '--n_classes',
        default=7,
        type=int,
        help="Number of segmentation classes"
    )
    parser.add_argument(
        '--reuse_model',
        default=0,
        type=int,
        help=""
    )
    parser.add_argument(
        '--batch_size',
        default=10,
        type=int,
        help=""
    )
    parser.add_argument(
        '--dropRate',
        default=0.0,
        type=float,
        help=""
    )
    parser.add_argument(
        '--num_epochs',
        default=30,
        type=int,
        help=""
    )
    parser.add_argument(
        '--initial_lr',
        default=0.1,
        type=float,
        help=""
    )
    parser.add_argument(
        '--milestones',
        default=[5,10,20],
        type=list,
        help=""
    )
    parser.add_argument(
        '--initial_label_weights',
        default=[0.00052994, 0.4663487 , 0.04557499, 0.09379968, 0.01748808, 0.10227875, 0.27397986],
        type=list,
        help=""
    )
    parser.add_argument(
        '--data_root',
        default='data/cache/train_test_data_npy_BiMask_sp1',
        type=str,
        help='Root directory path of data'
    )
    parser.add_argument(
        '--phase',
        default='train',
        type=str,
        help='train or test'
    )
    parser.add_argument(
        '--manual_seed',
        default=66,
        type=int,
        help=""
    )
    parser.add_argument(
        '--test_train_split_fpath',
        default='test_train_split.json',
        type=str,
        help='train or test'
    )
    parser.add_argument(
        '--gpu_id',
        nargs='+',
        type=int,
        help='Gpu id lists'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Gpu id lists'
    )
    parser.set_defaults(verbose=True)
    parser.add_argument(
        '--num_workers',
        default=6,
        type=int,
        help="num_workers for dataloaders"
    )
    args = parser.parse_args()
    return args