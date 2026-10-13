   
                                
   

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import os
import numpy as np
from tqdm import tqdm
import json
import random
import torch.nn.functional as F

from backbone_config import BackboneConfig
from backbone_model import DecoupledMoEBackbone
from backbone_dataset import create_backbone_dataloaders
from utils import AverageMeter, Logger


def set_seed(seed):
                                              
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_lr_schedule(optimizer, config, epoch):
                                                      
    if epoch < config.warmup_epochs:
                       
        lr = config.lr_warmup_start + (config.learning_rate - config.lr_warmup_start) * \
             (epoch / config.warmup_epochs)
    else:
                      
        progress = (epoch - config.warmup_epochs) / (config.max_epochs - config.warmup_epochs)
        lr = config.lr_min + (config.learning_rate - config.lr_min) * \
             0.5 * (1.0 + np.cos(np.pi * progress))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr


def save_checkpoint(state, checkpoint_dir, filename='checkpoint.pth'):
                         
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f"✓ Checkpoint saved: {filepath}")


def load_checkpoint(filepath, model, optimizer=None, device='cuda'):
                         
    if not os.path.exists(filepath):
        return None
    
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")
    return checkpoint


def compute_loss(pred, target, mask, weight_mask, aux_loss, config):
       
                      
    
         
                                        
                                           
                                   
                                           
                                                  
                              
        
            
                             
       
                                                    
    recon_loss = F.mse_loss(pred, target, reduction='none')                
    recon_loss = recon_loss.mean(dim=-1)           
    
                 
    mask_expanded = mask
    weight_expanded = weight_mask
    
                  
    masked_loss = recon_loss * mask_expanded * weight_expanded
    recon_loss_final = masked_loss.sum() / (mask_expanded * weight_expanded).sum().clamp(min=1)
    
                
    total_loss = config.lambda_recon * recon_loss_final + config.lambda_aux * aux_loss
    
    loss_dict = {
        'total': total_loss.item(),
        'recon': recon_loss_final.item(),
        'aux': aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
    }
    
    return total_loss, loss_dict


def train_one_epoch(model, train_loader, optimizer, scaler, config, epoch, logger):
                             
    model.train()
    
    loss_meter = AverageMeter()
    recon_meter = AverageMeter()
    aux_meter = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{config.max_epochs} [Train]')
    
    for step, batch in enumerate(pbar):
                        
        tokens = batch['tokens'].to(config.device)
        target = batch['target'].to(config.device)
        mask = batch['mask'].to(config.device)
        weight_mask = batch['weight_mask'].to(config.device)
        partition_ids = batch['partition_ids'].to(config.device)
        patch_ids = batch['patch_ids'].to(config.device)
        
                                                       
        with autocast(enabled=config.use_amp):
            pred, aux_loss, global_embedding = model(tokens, partition_ids, patch_ids, mask)
            loss, loss_dict = compute_loss(
                pred, target, mask, weight_mask, aux_loss, config
            )
        
                       
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
                           
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
                        
        batch_size = tokens.shape[0]
        loss_meter.update(loss_dict['total'], batch_size)
        recon_meter.update(loss_dict['recon'], batch_size)
        aux_meter.update(loss_dict['aux'], batch_size)
        
                             
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'recon': f'{recon_meter.avg:.4f}',
            'aux': f'{aux_meter.avg:.5f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
        
             
        if step % config.log_interval == 0:
            logger.log_train(epoch, step, {
                'total': loss_meter.avg,
                'recon': recon_meter.avg,
                'aux': aux_meter.avg,
                'lr': optimizer.param_groups[0]['lr']
            })
    
    return loss_meter.avg


@torch.no_grad()
def validate(model, val_loader, config, epoch, logger):
                            
    model.eval()
    
    loss_meter = AverageMeter()
    recon_meter = AverageMeter()
    aux_meter = AverageMeter()
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch}/{config.max_epochs} [Val]')
    
    for batch in pbar:
                        
        tokens = batch['tokens'].to(config.device)
        target = batch['target'].to(config.device)
        mask = batch['mask'].to(config.device)
        weight_mask = batch['weight_mask'].to(config.device)
        partition_ids = batch['partition_ids'].to(config.device)
        patch_ids = batch['patch_ids'].to(config.device)
        
                                                       
        with autocast(enabled=config.use_amp):
            pred, aux_loss, global_embedding = model(tokens, partition_ids, patch_ids, mask)
            loss, loss_dict = compute_loss(
                pred, target, mask, weight_mask, aux_loss, config
            )
        
                        
        batch_size = tokens.shape[0]
        loss_meter.update(loss_dict['total'], batch_size)
        recon_meter.update(loss_dict['recon'], batch_size)
        aux_meter.update(loss_dict['aux'], batch_size)
        
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'recon': f'{recon_meter.avg:.4f}',
            'aux': f'{aux_meter.avg:.5f}'
        })
    
                            
    logger.log_val(epoch, {
        'total': loss_meter.avg,
        'recon': recon_meter.avg,
        'aux': aux_meter.avg
    })
    
    return loss_meter.avg


def main():
                                
    config = BackboneConfig()
    
              
    set_seed(config.seed)
    
    print("="*80)
    print("MoE Backbone Training (MAE-Style)")
    print("="*80)
    print(f"Device: {config.device}")
    print(f"Encoder: {config.num_layers} layers, {config.num_experts} experts (layer 0)")
    print(f"Decoder: {config.decoder_num_layers} layers")
    print(f"Batch size: {config.batch_size}")
    print(f"Max epochs: {config.max_epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print("="*80)
    
                        
    print("\nLoading cached tokens...")
    train_loader, val_loader = create_backbone_dataloaders(config)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
                  
    print("\nInitializing model...")
    model = DecoupledMoEBackbone(config).to(config.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
               
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas
    )
    
                            
    scaler = GradScaler(enabled=config.use_amp)
    
            
    logger = Logger(config.log_dir)
    
                            
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    checkpoint_path = os.path.join(config.checkpoint_dir, 'latest.pth')
    checkpoint = load_checkpoint(checkpoint_path, model, optimizer, config.device)
    if checkpoint:
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        patience_counter = checkpoint.get('patience_counter', 0)
    
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")
    
                   
    for epoch in range(start_epoch, config.max_epochs):
                              
        current_lr = get_lr_schedule(optimizer, config, epoch)
        
               
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, config, epoch, logger
        )
        
                  
        if epoch % config.val_interval == 0:
            val_loss = validate(model, val_loader, config, epoch, logger)
            
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  LR: {current_lr:.6f}")
            print(f"  Best Val: {best_val_loss:.4f}")
            
                              
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'config': config.__dict__
                }, config.checkpoint_dir, 'best.pth')
                
                print("  *** New best model! ***")
            else:
                patience_counter += 1
                print(f"  Patience: {patience_counter}/{config.early_stopping_patience}")
            
                            
            if patience_counter >= config.early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
                                  
        if epoch % config.save_interval == 0:
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter,
                'config': config.__dict__
            }, config.checkpoint_dir, f'epoch_{epoch:03d}.pth')
        
                            
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
            'config': config.__dict__
        }, config.checkpoint_dir, 'latest.pth')
        
                   
        logger.save()
    
    print("\n" + "="*80)
    print("Training Complete!")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print("="*80)


if __name__ == '__main__':
    main()