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
    # parser.add_argument(
    #     '--reuse_model',
    #     default=0,
    #     type=int,
    #     help=""
    # )
    parser.add_argument(
        '--batch_size',
        required= True,
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
        required=True,
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
        nargs="*",
        type=int,
        default=[5,10,20],
        help=""
    )
    parser.add_argument(
        '--initial_label_weights',
        nargs="*",
        default=[0.00052994, 0.4663487 , 0.04557499, 0.09379968, 0.01748808, 0.10227875, 0.27397986],
        type=float,
        help=""
    )
    parser.add_argument(
        '--data_root',
        required=True,
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
        default=1,
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
        '--distributed',
        default=None,
        required=True,
        type=str,
        help='dp,ddp,None'
    )
    parser.add_argument(
        '--gpus',
        default= -1,
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
    parser.add_argument(
        '--debug',
        default=0,
        type=int,
        help=''
    )
    parser.add_argument(
        '--unet_type',
        required=True,
        default='3-1-3',
        type=str,
        help=''
    )
    parser.add_argument(
        '--cur_ckpt_loc',
        default= "/content/3DUnet/results",
        type=str,
        help=''
    )
    parser.add_argument(
        '--ID',
        default="DEFAULT-ID",
        required= True,
        type=str,
        help=''
    )
    parser.add_argument(
        '--save_additional_checkpoint',
        default="",
        required= True,
        type=str,
        help='save additional checkpoint and logging file to the path provided, always used when using Colab, used \
             to save mid point to Google Drive'
    )
    parser.add_argument(
        '--resume_path',
        default="",
        required=True,
        type=str,
        help='some/path/to/my_checkpoint.ckpt'
    )

    parser.add_argument(
        '--amp_level',
        default="O2",
        required=True,
        type=str,
        help='O0:FP32; O1,O2:Mixed; O3:FP16'
    )

    parser.add_argument(
        '--acc_grad',
        default=1,
        type=int,
        help="""accumulate every n batches (effective batch size is batch*n); 
        # no accumulation for epochs 1-4. accumulate 3 for epochs 5-10. accumulate 20 after that
        trainer = Trainer(accumulate_grad_batches={5: 3, 10: 20})"""
    )


    args = parser.parse_args()
    return args