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
print(opt)

use_cuda = torch.cuda.is_available() and not opt.no_cuda
if use_cuda:
    print("CUDA enabled.")

device = torch.device("cuda") if use_cuda else torch.device("cpu")

dataset_device = torch.device("cpu") if opt.dataset_main_memory else device

if opt.name is None:
    raise RuntimeError('Session name missing (--name) in config.')
    
if os.path.exists(os.path.join(opt.output_folder, opt.name + '.json')):
    raise RuntimeError(f'Session name {opt.name} taken.')

use_mixed_precision = opt.mixed_precision and amp.autocast_mode.is_autocast_available(device.type)
if use_mixed_precision:
    print("Mixed precision enabled.")

logger = logging.Logger(opt, print_epoch_metrics=True)

if opt.effective_batch_size % opt.batch_size != 0:
    raise ValueError("Effective batch size must be a multiple of batch size.")
accumulate_steps = opt.effective_batch_size // opt.batch_size

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

if opt.detail == 'target_only':
    detail_option = dataset.FeatureOptions.TARGET_ONLY
elif opt.detail == 'features_minimal':
    detail_option = dataset.FeatureOptions.FEATURES_MINIMAL
elif opt.detail == 'features_full':
    detail_option = dataset.FeatureOptions.FEATURES_FULL
else:
    assert False

graph_features = dataset.GraphFeatures(opt.dataset_folder, detail_option, device=dataset_device, store_half=True)
N, T, F = graph_features.features_tensor.shape  # nodes, times, features

print("Creating training masks...")

training_set = dataset.Dataset(graph_features, 
                               dataset.MaskSet(opt.training_masks, N, T, opt.noise_kernel_size, opt.unseen_split, device=dataset_device, seed=420),
                               opt.window_size, opt.batch_size)
validation_set = dataset.Dataset(graph_features, 
                                 dataset.MaskSet(opt.validation_masks, N, T, opt.noise_kernel_size, opt.unseen_split, device=dataset_device, seed=69),
                                 opt.window_size, opt.batch_size)

pin_memory = opt.dataset_main_memory and use_cuda
training_loader = DataLoader(training_set, batch_size=None, shuffle=True, pin_memory=pin_memory)
validation_loader = DataLoader(validation_set, batch_size=None, shuffle=False, pin_memory=pin_memory)

if use_cuda:
    torch.cuda.empty_cache()

# Model

graph = gnn.GCNGraph(graph_features.adjacency_matrix, device=device)

model = gnn.Model(
    gnn.GCNBlock((F+1) * opt.window_size, 64, graph),
    gnn.GCNBlock(64, 64, graph, residuals=True),
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


training_losses = torch.empty(len(training_loader), device=dataset_device)
validation_losses = torch.empty(len(validation_loader), device=dataset_device)    

for e in range(opt.max_epochs):

    training_set.reshuffle_offsets()

    model.train()
    optimizer.zero_grad()
    
    for i, (features, unseen_mask) in enumerate(tqdm(training_loader, f"Training epoch {e+1}")):

        with amp.autocast(device.type, enabled=use_mixed_precision):
            features = features.to(device=device, non_blocking=True)
            unseen_mask = unseen_mask.to(device=device, dtype=features.dtype, non_blocking=True)
            
            # add mask to model input
            features_all = torch.cat((unseen_mask.unsqueeze(-1), features), dim=-1)
            # mask target input variable
            features_all[:, :, :, 1] *= unseen_mask
            
            model_output = model(features_all)
            
            loss = loss_crit(model_output, features[:, :, :, 0], 1 - unseen_mask)
            
            training_losses[i] = loss
            loss = loss / accumulate_steps

        scaler.scale(loss).backward()

        if (i+1) % accumulate_steps == 0 or (i+1) == len(training_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    model.eval()

    with torch.no_grad():

        for i, (features, unseen_mask) in enumerate(tqdm(validation_loader, f"Validating epoch {e+1}")):
            features = features.to(device=device, non_blocking=True)
            unseen_mask = unseen_mask.to(device=device, dtype=features.dtype, non_blocking=True)
            
            features_all = torch.cat((unseen_mask.unsqueeze(-1), features), dim=-1)
            features_all[:, :, :, 1] *= unseen_mask
            
            model_output = model(features_all)
            
            loss = loss_crit(model_output, features[:, :, :, 0], 1 - unseen_mask)
            
            validation_losses[i] = loss

    avg_train_loss = training_losses.mean().item()
    avg_val_loss = validation_losses.mean().item()

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