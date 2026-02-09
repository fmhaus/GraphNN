import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from bach600 import config, dataset, gnn, utils

opt = config.get_config_options()

use_cuda = torch.cuda.is_available() and opt.use_gpu
device = torch.device("cuda") if use_cuda else torch.device("cpu")

print("Loading dataset...")

graph_features = dataset.GraphFeatures(opt.dataset_folder, device, dataset.FeatureOptions.TARGET_ONLY)
N, T, F = graph_features.features_tensor.shape  # nodes, times, features
H = opt.window_size

training_set = dataset.Dataset(graph_features, 
                               dataset.MaskSet(500, N, T, opt.noise_kernel_size, opt.unseen_split, device, seed=420),
                               opt.window_size)
validation_set = dataset.Dataset(graph_features, 
                                 dataset.MaskSet(20, N, T, opt.noise_kernel_size, opt.unseen_split, device, seed=69),
                                 opt.window_size)

training_loader = DataLoader(training_set, opt.batch_size, shuffle=True, pin_memory=use_cuda)
validation_loader = DataLoader(validation_set, opt.batch_size, shuffle=False, pin_memory=use_cuda)

print("Dataset loaded.")

graph = gnn.GCNGraph(dataset.adjacency_matrix, device=device)

model = gnn.Model(
    gnn.GCNBlock((F+1) * opt.window_size, 64, graph),
    gnn.GCNBlock(64, 64, graph),
    gnn.GCNBlock(64, 1 * opt.window_size, graph)
)
model.to(device)

loss_crit = torch.nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=opt.initial_lr)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.total_epochs, eta_min=1e-5)
early_stopping = utils.EarlyStopping(patience=20)

for e in range(opt.total_epochs):

    model.train()

    training_loss_sum = 0

    for features, mask in tqdm(training_loader, f"Training epoch {e+1}"):
        optimizer.zero_grad()

        print(features.shape)
        print(mask.shape)

        # Mask target and concat mask to features
        ground_truth = features[:, :, :, 0]
        features_masked = features[:, :, :, 0] * mask
        features_all = torch.cat((mask.unsqueeze(-1), features_masked), dim=-1)

        model_output = model(features_all)

        # Mask GT and loss for loss
        inverse_mask = 1 - mask
        model_output_masked = model_output * inverse_mask
        ground_truth_masked = ground_truth * inverse_mask
        
        loss = loss_crit(model_output_masked, ground_truth_masked)
        training_loss_sum += loss.item()

        # Do the learning
        loss.backward()
        optimizer.step()

    validation_loss_sum = 0

    with torch.no_grad():
        model.eval()

        for features, mask in tqdm(validation_loader, f"Validation epoch {e+1}"):
            ground_truth = features[:, :, :, 0]
            features_masked = features[:, :, :, 0] * mask
            features_all = torch.cat((mask.unsqueeze(-1), features_masked), dim=-1)

            model_output = model(features_all)

            inverse_mask = 1 - mask
            model_output_masked = model_output * inverse_mask
            ground_truth_masked = ground_truth * inverse_mask
            
            loss = loss_crit(model_output_masked, ground_truth_masked)
            validation_loss_sum += loss.item()

    avg_train_loss = training_loss_sum / len(training_set)
    avg_val_loss = validation_loss_sum / len(validation_set)

    scheduler.step()

    if early_stopping(loss.item()):
        break

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {e+1} completed. Training loss: {avg_train_loss}. Validation loss: {avg_val_loss}. LR: {current_lr}")