import argparse

def get_config_options():
    parser = argparse.ArgumentParser()

    # train parameters
    parser.add_argument('--total_epochs', type=int, default=300, help='toal number of epochs')
    parser.add_argument('--initial_lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--gpu', type=bool, default=True, help='Whether to use gpu acceleration (if available)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--dataset_folder', type=str, default='dataset/', help='Dataset folder')
    
    parser.add_argument('--window_size', type=int, default=10, help='Temporal window size')
    parser.add_argument('--unseen_split', type=float, default=0.1, help='The split of seen/unseen sensors')
    parser.add_argument('--noise_kernel_size', type=float, default=5, help='The kernel size for random artificial sensor loss')

    return parser.parse_args()
