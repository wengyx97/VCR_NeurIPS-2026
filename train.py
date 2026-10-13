   
                                             

                
                                                           
                                                       
   

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import os
import time
from tqdm import tqdm

from config import Config
from dataset import create_dataloaders
from models import MultiModalTokenizer
from losses import TotalLoss
from utils import (AverageMeter, save_checkpoint, load_checkpoint, 
                   get_lr_schedule, get_contrastive_weight, Logger,
                   visualize_reconstruction, plot_rank_evolution)


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, 
                   config, epoch, logger):
       
                       
    
         
                              
                                         
                                
                            
                                              
                                    
                                   
                             
        
            
                                       
       
    model.train()
    
    loss_meter = AverageMeter()
    recon_meter = AverageMeter()
    contrastive_meter = AverageMeter()
    white_meter = AverageMeter()                             
    
                                     
    lambda_contrastive = get_contrastive_weight(config, epoch)
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{config.max_epochs} [Train]')
    
    for step, batch in enumerate(pbar):
                              
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(config.device)
        
                                          
        targets = {
            'ppg': batch['ppg'],
            'eda': batch['eda'],
            'acc': batch['acc'],
            'temp': batch['temp']
        }
        
                                           
        with autocast():
            outputs = model(batch)
                                                   
            loss, loss_dict = criterion(outputs, targets, 
                                       lambda_recon=config.lambda_recon,
                                       lambda_contrastive=lambda_contrastive,
                                       lambda_white=config.lambda_white)
        
                       
        optimizer.zero_grad()
        scaler.scale(loss).backward()

                                             
                                          
                              
                                                                                  
            
                                        
                                                                
                                   
                                                    
                
                          
                                             
                                                                                      
                     
                                    
                                                                     
                                                                   
                    
                                                 
                                                               
                    
                                                                                                                                   
            
                                                                               
                                             
        
        scaler.step(optimizer)
        scaler.update()
        
                        
        batch_size = batch['ppg'].shape[0]
        loss_meter.update(loss.item(), batch_size)
        recon_meter.update(loss_dict['recon_total'].item(), batch_size)
        contrastive_meter.update(loss_dict['contrastive'].item(), batch_size)
        white_meter.update(loss_dict['white'].item(), batch_size)       
        ranks_info = {k: v for k, v in outputs['current_ranks'].items()}
                             
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'recon': f'{recon_meter.avg:.4f}',
            'contr': f'{contrastive_meter.avg:.4f}',
            'white': f'{white_meter.avg:.5f}',       
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}',
            'R_ppg': f'{ranks_info["ppg"]:.1f}', 
            'R_eda': f'{ranks_info["eda"]:.1f}',
            'R_acc': f'{ranks_info["acc"]:.1f}',
            'R_temp': f'{ranks_info["temp"]:.1f}',
        })
        
                       
        if step % config.log_interval == 0:
            log_payload = {
                'total': loss_meter.avg,
                'recon_total': recon_meter.avg,
                'contrastive': contrastive_meter.avg,
                'white': white_meter.avg,
                'lr': optimizer.param_groups[0]['lr'],
                            
                'rank_ppg': ranks_info['ppg'],
                'rank_eda': ranks_info['eda'],
                'rank_acc': ranks_info['acc'],
                'rank_temp': ranks_info['temp']
            }
            logger.log_train(epoch, step, log_payload)
    
    return loss_meter.avg


@torch.no_grad()
def validate(model, val_loader, criterion, config, epoch, logger, save_vis=False):
       
                      
    
         
                              
                                         
                                
                                    
                                   
                             
                                               
        
            
                                         
       
    model.eval()
    
    loss_meter = AverageMeter()
    recon_meter = AverageMeter()
    contrastive_meter = AverageMeter()
    white_meter = AverageMeter()                             
    
                                        
    recon_meters = {
        'ppg': AverageMeter(),
        'eda': AverageMeter(),
        'acc': AverageMeter(),
        'temp': AverageMeter()
    }
    
    lambda_contrastive = get_contrastive_weight(config, epoch)
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch}/{config.max_epochs} [Val]')
    
    first_batch = None
    
    for step, batch in enumerate(pbar):
                                            
        if first_batch is None:
            first_batch = batch
        
                              
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(config.device)
        
                         
        targets = {
            'ppg': batch['ppg'],
            'eda': batch['eda'],
            'acc': batch['acc'],
            'temp': batch['temp']
        }
        
                      
        with autocast():
            outputs = model(batch)
                                                   
            loss, loss_dict = criterion(outputs, targets,
                                       lambda_recon=config.lambda_recon,
                                       lambda_contrastive=lambda_contrastive,
                                       lambda_white=config.lambda_white)
        
                        
        batch_size = batch['ppg'].shape[0]
        loss_meter.update(loss.item(), batch_size)
        recon_meter.update(loss_dict['recon_total'].item(), batch_size)
        contrastive_meter.update(loss_dict['contrastive'].item(), batch_size)
        white_meter.update(loss_dict['white'].item(), batch_size)       
        
        for mod in recon_meters.keys():
            recon_meters[mod].update(loss_dict[f'recon_{mod}'].item(), batch_size)
        
                             
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'recon': f'{recon_meter.avg:.4f}',
            'contr': f'{contrastive_meter.avg:.4f}',
            'white': f'{white_meter.avg:.5f}'       
        })
    
                            
    log_dict = {
        'total': loss_meter.avg,
        'recon_total': recon_meter.avg,
        'contrastive': contrastive_meter.avg,
        'white': white_meter.avg,       
    }
    for mod in recon_meters.keys():
        log_dict[f'recon_{mod}'] = recon_meters[mod].avg
    
    logger.log_val(epoch, log_dict)
    
                                       
    if save_vis and first_batch is not None:
        vis_dir = os.path.join(config.log_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
                                                    
        for key in first_batch:
            if isinstance(first_batch[key], torch.Tensor):
                first_batch[key] = first_batch[key].to(config.device)
        
        with autocast():
            vis_outputs = model(first_batch)
        
                                                
        for mod in ['ppg', 'eda', 'acc', 'temp']:
            save_path = os.path.join(vis_dir, f'epoch_{epoch:03d}_{mod}.png')
            visualize_reconstruction(
                first_batch[mod][0],                
                vis_outputs['reconstructions'][mod][0],
                mod,
                save_path
            )
    
    return loss_meter.avg


def main():
                                
    
                   
    config = Config()
    
    print("="*80)
    print("Multi-Modal Tokenizer Training")
    print("  [UPDATED: Orthogonal Projection + Whitening Regularization]")
    print("="*80)
    print(f"Device: {config.device}")
    print(f"Total samples: {config.total_samples}")
    print(f"Batch size: {config.batch_size}")
    print(f"Max epochs: {config.max_epochs}")
    print(f"d_model: {config.d_model}")
    print(f"Lambda recon: {config.lambda_recon}")
    print(f"Lambda contrastive: {config.lambda_contrastive_start} -> {config.lambda_contrastive_end}")
    print(f"Lambda whitening: {config.lambda_white}")       
    print("="*80)
    
                                               
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        config,
        use_cache=True,                               
        force_reload=False,                                              
        num_processes=None                                    
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Actual train_loader workers: {train_loader.num_workers}")
    print(f"Actual val_loader workers: {val_loader.num_workers}")
    
                  
    print("\nInitializing model...")
    model = MultiModalTokenizer(d_model=config.d_model, init_rank_ratio=config.init_rank_ratio).to(config.device)
    
                      
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

                                                
             
                                           
                                                         
                      
    
    gate_params = []
    matrix_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if 'gate_logits' in name:
            gate_params.append(param)
            print(f"  > Group [Gate]: {name} (High LR)")
        elif 'W' in name and 'projector' in name: 
                                  
            matrix_params.append(param)
            print(f"  > Group [Matrix]: {name} (Low LR - Stabilized)")
        else:
            other_params.append(param)
    
           
                                                            
                                
    optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': config.learning_rate},                     
        {'params': gate_params,  'lr': config.learning_rate * 10},               
        {'params': matrix_params,'lr': config.learning_rate * 0.01}                         
    ], weight_decay=config.weight_decay)
    
                                               
                                      
                                                                                          
                                                
    
                      
                                     
                           
                                
                                        
        
    
                          
                                                          
    criterion = TotalLoss(
        temperature=config.temperature,
        num_patches=config.num_patches                      
    )
                            
    scaler = GradScaler()
    
                   
    logger = Logger(config.log_dir)
    
                             
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
                                      
    checkpoint_path = os.path.join(config.checkpoint_dir, 'latest.pth')
    if os.path.exists(checkpoint_path):
        print(f"\nResuming from checkpoint: {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path, model, optimizer, config.device)
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        patience_counter = checkpoint.get('patience_counter', 0)
    
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")
    
                   
    for epoch in range(start_epoch, config.max_epochs):
                              
        current_lr = get_lr_schedule(optimizer, config, epoch)
        
                         
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            config, epoch, logger
        )
        
                  
        if epoch % config.val_interval == 0:
            val_loss = validate(
                model, val_loader, criterion, config, epoch, logger,
                save_vis=(epoch % (config.val_interval * 5) == 0)                               
            )
            
            print(f"\nEpoch {epoch}/{config.max_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            print(f"  Best Val Loss: {best_val_loss:.4f}")
            
                                 
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
                
                print(f"  *** New best model saved! ***")
            else:
                patience_counter += 1
                print(f"  Patience: {patience_counter}/{config.early_stopping_patience}")
            
                            
            if patience_counter >= config.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
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
        
        print()
    
                   
    print("\n" + "="*80)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*80)
    
                 
    logger.plot_losses(os.path.join(config.log_dir, 'final_losses.png'))
    plot_rank_evolution(config.log_dir)


if __name__ == '__main__':
    main()