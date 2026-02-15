import random
import os
import time
import torch
from torch.utils.data import DataLoader
from torch import amp
import numpy as np
from tqdm import tqdm

from bach600 import config, dataset, gnn, utils, logging, loss

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

use_mixed_precision = opt.mixed_precision and amp.autocast_mode.is_autocast_available(device.type)
if use_mixed_precision:
    print("Mixed precision enabled.")

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

graph_features = dataset.GraphFeatures(opt.dataset_folder, dataset.FeatureOptions.TARGET_ONLY, store_half=True)
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

if not opt.no_compile:
    model = torch.compile(model)
    print("Compiling model.")

loss_crit = loss.MaskedMAE()
optimizer = torch.optim.Adam(model.parameters(), lr=opt.initial_lr)

scaler = amp.GradScaler(enabled=use_mixed_precision)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3,
    min_lr=1e-5,
    threshold=1e-4
)

early_stopping = utils.EarlyStopping(patience=10, min_delta=1e-4)

start_time = time.time()

for e in range(opt.max_epochs):

    training_loss_sum = 0
    model.train()
    
    for features, unseen_mask in tqdm(training_loader, f"Training epoch {e+1}"):
        
        optimizer.zero_grad()

        with amp.autocast(device.type, enabled=use_mixed_precision):
            if features.device == device:
                features = features.clone()
            else:
                features = features.to(device=device, non_blocking=True)
            
            unseen_mask = unseen_mask.to(device=device, dtype=features.dtype, non_blocking=True)

            # Mask target and concat mask to features
            ground_truth = features[:, :, :, 0].clone()
            features[:, :, :, 0] *= unseen_mask

            features_all = torch.cat((unseen_mask.unsqueeze(-1), features), dim=-1)

            model_output = model(features_all)

            seen_mask = 1 - unseen_mask
            loss = loss_crit(model_output, ground_truth, seen_mask)
            training_loss_sum += loss.item()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    validation_loss_sum = 0
    model.eval()

    with torch.no_grad():

        for features, unseen_mask in tqdm(validation_loader, f"Validating epoch {e+1}"):
            if features.device == device:
                features = features.clone()
            else:
                features = features.to(device=device, non_blocking=True)
            
            unseen_mask = unseen_mask.to(device=device, dtype=torch.float, non_blocking=True)
            
            ground_truth = features[:, :, :, 0].clone()
            features[:, :, :, 0] *= unseen_mask
            features_all = torch.cat((unseen_mask.unsqueeze(-1), features), dim=-1)

            model_output = model(features_all)
            
            seen_mask = 1 - unseen_mask
            loss = loss_crit(model_output, ground_truth, seen_mask)
            validation_loss_sum += loss.item()

    avg_train_loss = training_loss_sum / len(training_loader)
    avg_val_loss = validation_loss_sum / len(validation_loader)

    current_lr = optimizer.param_groups[0]['lr']
    logger.log_epoch(model, e, avg_val_loss, {
        'learning_rate': current_lr, 
        'average_training_loss': avg_train_loss, 
        'average_validation_loss': avg_val_loss, 
        'average_multiplicative_error': graph_features.get_multiplicative_error(avg_val_loss)
    })

    scheduler.step(avg_val_loss)
    
    if early_stopping(avg_val_loss):
        break

print(f"Passed time: {time.time() - start_time}")

logger.save()
print("Saved results.")