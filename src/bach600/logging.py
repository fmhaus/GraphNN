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
        self.best_loss = float('inf')
    
    def log_epoch(self, model, index: int, loss: float, parameters):
        self.data["epochs"].append({"index": index} | parameters)
        self.data["last_epoch"] = index
        
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_state = model.state_dict()
            self.data["best_epoch"] = index
        
        if self.print_epoch_metrics:
            print(f"Epoch {index+1} completed: {parameters}")
        
        if OPEN_BAYES:        
            for k, v in parameters.items():
                if index == 0:
                    openbayestool.clear_metric(k)
                
                openbayestool.log_metric(k, v)
    
    def save(self):
        with open(os.path.join(self.folder, self.name + ".json"), "w") as fr:
            json.dump(self.data, fr)
        if self.best_state:
            torch.save(self.best_state, os.path.join(self.folder, self.name + ".pth"))