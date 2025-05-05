import argparse
import os

def init_args(args):
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    return args

def parse_args():
    descript = 'Pytorch Implementation of UR-DMU'
    parser = argparse.ArgumentParser(description=descript)

    parser.add_argument('--output_path', type=str, default='outputs/')
    parser.add_argument('--root_dir', type=str, default='outputs/')
    parser.add_argument('--log_path', type=str, default='logs/')

    # Type of features to use
    parser.add_argument('--modal', type=str, default='rgb', choices=["rgb", "flow", "both"])

    # Path to saved models
    parser.add_argument('--model_path', type=str, default='models/')

    # Path to feature directory (optional but useful)
    parser.add_argument('--feature_path', type=str,
                        default='/mnt/d/UR-DMU/i3d-features/i3d-features/rgb/',
                        help='Path to the directory with I3D RGB features')

    parser.add_argument('--lr', type=str, default='[0.0001]*3000', help='learning rates for steps (list form)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_segments', type=int, default=32)
    parser.add_argument('--seed', type=int, default=2022, help='random seed (-1 for no manual seed)')

    # ✅ Fixed: model file defaults to correct name
    parser.add_argument('--model_file', type=str, default='ucf_trans_2022.pkl',
                        help='Path to pretrained model file')

    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    return init_args(args)
