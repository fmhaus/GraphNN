import os
import json
import torch

try:
    import openbayestool
    OPEN_BAYES = True
except:
    OPEN_BAYES = False

class Logger:
    def __init__(self, opt, print_epoch_metrics: bool = False):
        self.folder = opt.output_folder
        self.name = opt.name
        self.print_epoch_metrics = print_epoch_metrics
        
        self.data = {
            "epochs": [],
            "last_epoch": -1,
            "best_epoch": -1,
            "options": {
                "max_epochs": opt.max_epochs,
                "initial_lr": opt.initial_lr,
                "batch_size": opt.batch_size,
                "training_masks": opt.training_masks,
                "validation_masks": opt.validation_masks,
                "window_size": opt.window_size,
                "unseen_split": opt.unseen_split,
                "noise_kernel_size": opt.noise_kernel_size
            }
        }
        
        self.best_state = None
        self.best_val = float('inf')
        
        if OPEN_BAYES:
            openbayestool.clear_metric("learning rate")
            openbayestool.clear_metric("training loss")
            openbayestool.clear_metric("validation loss")
            openbayestool.clear_param("epoch")
    
    def log_epoch(self, model, index: int, learning_rate: float, train_loss: float, val_loss: float):
        self.data["epochs"].append({
            "index": index,
            "learning_rate": learning_rate,
            "train_loss": train_loss,
            "val_loss": val_loss
        })
        self.data["last_epoch"] = index
        
        if val_loss < self.best_val:
            self.best_val = val_loss
            self.best_state = model.state_dict()
            self.data["best_epoch"] = index
        
        if self.print_epoch_metrics:
            print(f"Epoch {index+1} completed. Training loss: {train_loss}. Validation loss: {val_loss}. LR: {learning_rate}")
        
        if OPEN_BAYES:
            openbayestool.log_metric("learning rate", learning_rate)
            openbayestool.log_metric("training loss", train_loss)
            openbayestool.log_metric("validation loss", val_loss)
            openbayestool.log_param("epoch", index+1)
    
    def save(self):
        with open(os.path.join(self.folder, self.name + ".json"), "w") as fr:
            json.dump(self.data, fr)
        if self.best_state:
            torch.save(self.best_state, os.path.join(self.folder, self.name + ".pth"))