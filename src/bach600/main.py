import random, os

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from bach600 import config, dataset, gnn, utils, logging

# Config

opt = config.get_config_options()

use_cuda = torch.cuda.is_available() and not opt.no_cuda
if use_cuda:
    print("CUDA enabled.")

device = torch.device("cuda") if use_cuda else torch.device("cpu")

if opt.name is None:
    raise RuntimeError('Session name missing (--name) in config.')
    
if os.path.exists(os.path.join(opt.output_folder, opt.name + '.json')):
    raise RuntimeError(f'Session name {opt.name} taken.')

logger = logging.Logger(opt, print_epoch_metrics=True)

# Reproducability

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(SEED)

# Dataset

print("Loading dataset...")

graph_features = dataset.GraphFeatures(opt.dataset_folder, dataset.FeatureOptions.TARGET_ONLY)
N, T, F = graph_features.features_tensor.shape  # nodes, times, features

print("Creating training masks...")

training_set = dataset.Dataset(graph_features, 
                               dataset.MaskSet(opt.training_masks, N, T, opt.noise_kernel_size, opt.unseen_split, seed=420),
                               opt.window_size)
validation_set = dataset.Dataset(graph_features, 
                                 dataset.MaskSet(opt.validation_masks, N, T, opt.noise_kernel_size, opt.unseen_split, seed=69),
                                 opt.window_size)

training_loader = DataLoader(training_set, opt.batch_size, shuffle=True, pin_memory=use_cuda, collate_fn=dataset.concat_collate)
validation_loader = DataLoader(validation_set, opt.batch_size, shuffle=False, pin_memory=use_cuda, collate_fn=dataset.concat_collate)

# Model

graph = gnn.GCNGraph(graph_features.adjacency_matrix, device=device)

model = gnn.Model(
    gnn.GCNBlock((F+1) * opt.window_size, 64, graph),
    gnn.GCNBlock(64, 64, graph),
    gnn.GCNBlock(64, 1 * opt.window_size, graph)
)
model = model.to(device)

if opt.compile:
    model = torch.compile(model)
    print("Compiling model.")

loss_crit = torch.nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=opt.initial_lr)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.75,
    patience=3,
    min_lr=1e-6
)

early_stopping = utils.EarlyStopping(patience=10, min_delta=5e-4)

for e in range(opt.max_epochs):

    training_loss_sum = 0
    model.train()
    
    torch.cuda.empty_cache()
    
    for features, mask in tqdm(training_loader, f"Training epoch {e+1}"):
        
        optimizer.zero_grad()

        features = features.clone().to(device=device)
        mask = mask.to(device=device, dtype=torch.float)
        
        # Mask target and concat mask to features
        ground_truth = features[:, :, :, 0].clone()
        features[:, :, :, 0] *= mask
        
        features_all = torch.cat((mask.unsqueeze(-1), features), dim=-1)

        model_output = model(features_all)

        # Mask GT and loss for loss
        inverse_mask = 1 - mask
        model_output_masked = model_output * inverse_mask
        ground_truth_masked = ground_truth * inverse_mask
        
        assert model_output_masked.shape == ground_truth_masked.shape, f"{model_output_masked.shape} vs {ground_truth_masked.shape}"
        assert model_output_masked.device == ground_truth_masked.device
        assert model_output_masked.dtype == ground_truth_masked.dtype
        assert not torch.isnan(model_output_masked).any()
        assert not torch.isnan(ground_truth_masked).any()
        
        loss = loss_crit(model_output_masked, ground_truth_masked)
        training_loss_sum += loss.item()

        # Do the learning
        loss.backward()
        optimizer.step()

    validation_loss_sum = 0
    model.eval()

    with torch.no_grad():

        for features, mask in tqdm(validation_loader, f"Validating epoch {e+1}"):
            features = features.clone().to(device=device)
            mask = mask.to(device=device, dtype=torch.float)
            
            ground_truth = features[:, :, :, 0].clone()
            features[:, :, :, 0] *= mask
            features_all = torch.cat((mask.unsqueeze(-1), features), dim=-1)

            model_output = model(features_all)

            inverse_mask = 1 - mask
            model_output_masked = model_output * inverse_mask
            ground_truth_masked = ground_truth * inverse_mask
            
            loss = loss_crit(model_output_masked, ground_truth_masked)
            validation_loss_sum += loss.item()

    avg_train_loss = training_loss_sum / len(training_loader)
    avg_val_loss = validation_loss_sum / len(validation_loader)

    current_lr = optimizer.param_groups[0]['lr']
    logger.log_epoch(model, e, current_lr, avg_train_loss, avg_val_loss)
    
    print(f"Average multiplicate error factor: {dataset.features.interpret_target_error(avg_val_loss)}")

    scheduler.step(avg_val_loss)
    
    if early_stopping(avg_val_loss):
        break
        
logger.save()
print("Saved results.")