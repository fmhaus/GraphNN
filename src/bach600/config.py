import argparse

def get_config_options():
    parser = argparse.ArgumentParser()

    # train parameters
    parser.add_argument('--max_epochs', type=int, default=1000, help='The maximum number of epochs to train for')
    parser.add_argument('--initial_lr', type=float, default=5e-3, help='Initial learning rate')
    parser.add_argument('--gpu', type=bool, default=True, help='Whether to use gpu acceleration (if available)')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size for training')
    parser.add_argument('--compile', action='store_true', default='False', help='Whether to compile the model')
    
    parser.add_argument('--training_masks', type=int, default=402, help='The amount of unique masks for training')
    parser.add_argument('--validation_masks', type=int, default=30, help='The amount of unique masks for validation')
    parser.add_argument('--window_size', type=int, default=10, help='Temporal window size')
    parser.add_argument('--unseen_split', type=float, default=0.1, help='The split of seen/unseen sensors')
    parser.add_argument('--noise_kernel_size', type=float, default=5, help='The kernel size for random artificial sensor loss')
    
    parser.add_argument('--dataset_folder', type=str, default='dataset/', help='Dataset folder')
    parser.add_argument('--output_folder', type=str, default='output/', help='Where to save model logs and states')
    parser.add_argument('--name', type=str, help='Filename of state and log saves')
    
    return parser.parse_args()
