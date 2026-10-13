   
                              
   

import torch
import os
import json
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
from config import Config

class AverageMeter:
                                                           
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state: Dict, checkpoint_dir: str, filename: str = 'checkpoint.pth'):
       
                            
    
         
                                                                       
                                                    
                                     
       
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath: str, model, optimizer=None, device='cuda'):
       
                            
    
         
                                         
                                       
                                                        
                                         
        
            
                                                        
       
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {filepath}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Best Val Loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")
    
    return checkpoint


                                                              
       
                                                                    
    
         
                            
                                    
                                           
       
                                            
                       
                                                                                         
                                                   
         
                      
                                                                                                      
                                                                       
                                                   
    
                                              
                              
    
                

def get_lr_schedule(optimizer, config, current_epoch: int):
       
                                                                    
                                                                            
       
                              
    if current_epoch < config.warmup_epochs:
                       
        base_lr = config.lr_warmup_start + (config.learning_rate - config.lr_warmup_start) * \
             (current_epoch / config.warmup_epochs)
    else:
                      
        progress = (current_epoch - config.warmup_epochs) / (config.max_epochs - config.warmup_epochs)
        base_lr = config.lr_min + (config.learning_rate - config.lr_min) * \
             0.5 * (1.0 + np.cos(np.pi * progress))
    
                    
                                                                            
                                                     
    
    for param_group in optimizer.param_groups:
                                
        if 'initial_lr' not in param_group:
            param_group['initial_lr'] = param_group['lr']
            
                                                
                                                             
                                            
        
                                                
                                                            
        
                           
        ratio = base_lr / config.learning_rate
        param_group['lr'] = param_group['initial_lr'] * ratio
    
    return base_lr


def get_contrastive_weight(config, current_epoch: int):
       
                                                  
    
         
                                    
                                           
        
            
                                                       
       
    if current_epoch < config.contrastive_warmup_epochs:
                                             
        progress = current_epoch / config.contrastive_warmup_epochs
        lambda_contrastive = config.lambda_contrastive_start + \
                           (config.lambda_contrastive_end - config.lambda_contrastive_start) * progress
    else:
        lambda_contrastive = config.lambda_contrastive_end
    
    return lambda_contrastive


class Logger:
                                            
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.train_log = []
        self.val_log = []
    
    def log_train(self, epoch: int, step: int, metrics: Dict):
                                  
        entry = {'epoch': epoch, 'step': step, **metrics}
        self.train_log.append(entry)
    
    def log_val(self, epoch: int, metrics: Dict):
                                    
        entry = {'epoch': epoch, **metrics}
        self.val_log.append(entry)
    
    def save(self):
                                     
        with open(os.path.join(self.log_dir, 'train_log.json'), 'w') as f:
            json.dump(self.train_log, f, indent=2)
        
        with open(os.path.join(self.log_dir, 'val_log.json'), 'w') as f:
            json.dump(self.val_log, f, indent=2)
    
    def plot_losses(self, save_path: str = None):
                                                 
        if len(self.val_log) == 0:
            return
        
        epochs = [entry['epoch'] for entry in self.val_log]
        val_total = [entry['total'] for entry in self.val_log]
        val_recon = [entry['recon_total'] for entry in self.val_log]
        val_contrastive = [entry['contrastive'] for entry in self.val_log]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
                    
        axes[0].plot(epochs, val_total, 'o-', label='Validation')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Total Loss')
        axes[0].set_title('Total Loss')
        axes[0].legend()
        axes[0].grid(True)
        
                             
        axes[1].plot(epochs, val_recon, 'o-', label='Validation', color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Reconstruction Loss')
        axes[1].set_title('Reconstruction Loss')
        axes[1].legend()
        axes[1].grid(True)
        
                          
        axes[2].plot(epochs, val_contrastive, 'o-', label='Validation', color='orange')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Contrastive Loss')
        axes[2].set_title('Contrastive Loss')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Loss plots saved to {save_path}")
        else:
            plt.savefig(os.path.join(self.log_dir, 'losses.png'), dpi=150, bbox_inches='tight')
        
        plt.close()


def visualize_reconstruction(original: torch.Tensor, reconstructed: torch.Tensor, 
                            modality: str, save_path: str):
       
                                                                  
    
         
                                                        
                                                                  
                               
                                      
       
    original = original.detach().cpu().numpy()
    reconstructed = reconstructed.detach().cpu().numpy()
    
    num_patches = original.shape[0]
    num_channels = original.shape[1]
    
    fig, axes = plt.subplots(num_channels, 1, figsize=(15, 3 * num_channels))
    if num_channels == 1:
        axes = [axes]
    
    for ch in range(num_channels):
                                               
        orig_signal = np.concatenate([original[i, ch, :] for i in range(num_patches)])
        recon_signal = np.concatenate([reconstructed[i, ch, :] for i in range(num_patches)])
        
        axes[ch].plot(orig_signal, label='Original', alpha=0.7)
        axes[ch].plot(recon_signal, label='Reconstructed', alpha=0.7)
        axes[ch].set_title(f'{modality.upper()} - Channel {ch}')
        axes[ch].set_xlabel('Time')
        axes[ch].set_ylabel('Amplitude')
        axes[ch].legend()
        axes[ch].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Reconstruction visualization saved to {save_path}")

          

def plot_rank_evolution(log_dir: str, save_path: str = None):
       
                                               
       
    import os
    import json
    import matplotlib.pyplot as plt
    config = Config()

    json_path = os.path.join(log_dir, 'train_log.json')
    if not os.path.exists(json_path):
        print(f"No log file found at {json_path}, skipping rank plot.")
        return

          
    with open(json_path, 'r') as f:
        logs = json.load(f)

                                 
    latest_entry_per_epoch = {}

    for entry in logs:
        if 'rank_ppg' in entry:
            epoch = entry['epoch']
                               
            latest_entry_per_epoch[epoch] = entry

                
    sorted_epochs = sorted(latest_entry_per_epoch.keys())

            
    ranks = {'ppg': [], 'eda': [], 'acc': [], 'temp': []}
    for mod in ranks:
        ranks[mod].append(int(config.d_model * config.init_rank_ratio))
    for epoch in sorted_epochs:
        entry = latest_entry_per_epoch[epoch]
        for mod in ranks:
            ranks[mod].append(entry.get(f'rank_{mod}', 0))

        
    plt.figure(figsize=(12, 6))
    colors = {'ppg': 'red', 'eda': 'blue', 'acc': 'green', 'temp': 'orange'}

    for mod, values in ranks.items():
        plt.plot(sorted_epochs, values, label=mod.upper(), color=colors.get(mod, 'black'), linewidth=2)

    plt.title('Evolution of Shared Rank per Modality (Last per Epoch)')
    plt.xlabel('Epoch')
    plt.ylabel('Effective Rank')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

          
    if save_path is None:
        save_path = os.path.join(log_dir, 'rank_evolution_epoch.png')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Rank evolution plot saved to {save_path}")