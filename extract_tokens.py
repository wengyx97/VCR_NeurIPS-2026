   
                                                      
                                                               
   

import torch
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path

from config import Config
from dataset import create_dataloaders
from models import MultiModalTokenizer
from utils import load_checkpoint


class TokenExtractor:
       
                                                               
    
           
                                                                           
                                                                             
       
    
    def __init__(self, config, checkpoint_path, device='cuda'):
        self.config = config
        self.device = device
        
                                   
        print("Loading pretrained Tokenizer...")
        self.model = MultiModalTokenizer(
            d_model=config.d_model,
            init_rank_ratio=config.init_rank_ratio
        ).to(device)
        
        checkpoint = load_checkpoint(checkpoint_path, self.model, device=device)
        self.model.eval()
        print(f"Tokenizer loaded from epoch {checkpoint.get('epoch', 'N/A')}")
    
    @torch.no_grad()
    def extract_batch(self, batch):
           
                                                 
        
                
                                     
                                     
           
                        
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
        
                      
        outputs = self.model(batch)
        
        B = batch['ppg'].shape[0]
        modalities = ['ppg', 'eda', 'acc', 'temp']
        
        S_list = []
        T_list = []
        
        for mod in modalities:
                                                               
                                                                          
                                                                              
            
                                                             
            h_white = outputs['patch_proj_inputs'][mod]                   
                                                                                    
            
                                 
            S_mod = outputs['patch_shared_embeddings'][mod].cpu()
            T_mod = outputs['patch_specific_embeddings'][mod]
                                                            
            
            S_list.append(S_mod)
            T_list.append(T_mod)
        
                                
        S_tokens = torch.stack(S_list, dim=1)
        T_tokens = torch.stack(T_list, dim=1)
        
        return S_tokens.cpu(), T_tokens.cpu()
    
    def extract_and_save(self, dataloader, save_path, split_name='train'):
           
                                                              
        
             
                                  
                                            
                                        
           
        print(f"\nExtracting {split_name} tokens...")
        
                             
        S_tokens_list = []
        T_tokens_list = []
        
        for batch in tqdm(dataloader, desc=f'Extracting {split_name}'):
            S_batch, T_batch = self.extract_batch(batch)
            S_tokens_list.append(S_batch)
            T_tokens_list.append(T_batch)
        
                     
        S_tokens = torch.cat(S_tokens_list, dim=0)                   
        T_tokens = torch.cat(T_tokens_list, dim=0)
        
                                        
        print(f"Converting to float16...")
        S_tokens_fp16 = S_tokens.half()
        T_tokens_fp16 = T_tokens.half()
        
                          
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        torch.save({
            'S_tokens': S_tokens_fp16,
            'T_tokens': T_tokens_fp16,
            'num_samples': S_tokens.shape[0],
            'd_model': self.config.d_model,
            'num_modalities': 4,
            'num_patches': 10,
            'modalities': ['ppg', 'eda', 'acc', 'temp'],
            'dtype': 'float16'
        }, save_path)
        
        print(f"✓ Saved to {save_path}")
        print(f"  Total samples: {S_tokens.shape[0]}")
        
                     
        self._verify_file(save_path)
    
    def _verify_file(self, filepath):
                                   
        data = torch.load(filepath, map_location='cpu', weights_only=False)
        
                
        for i in [0, 100, 500]:
            S_sample = data['S_tokens'][i, 0, 0].float()
            T_sample = data['T_tokens'][i, 0, 0].float()
            
            S_norm = torch.norm(S_sample).item()
            T_norm = torch.norm(T_sample).item()
            
            print(f"Sample {i}:")
            print(f"  S norm: {S_norm:.4f} (should ≈ 1)")
            print(f"  T norm: {T_norm:.4f} (should ≈ 1)")



def main():
                                  
    config = Config()
    
           
    checkpoint_path = 'checkpoints_good_tokenizer/tokenizer.pth'
    cache_dir = 'tmp/backbone_tokens'
    
    train_save_path = os.path.join(cache_dir, 'train_tokens.pt')
    val_save_path = os.path.join(cache_dir, 'val_tokens.pt')
    
    print("="*80)
    print("Token Extraction Pipeline")
    print("="*80)
    print(f"Tokenizer checkpoint: {checkpoint_path}")
    print(f"Output directory: {cache_dir}")
    print(f"Output format: .pt (float16)")
    print("="*80)
    
                             
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
                        
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        config,
        use_cache=True,
        force_reload=False
    )
    
                          
    extractor = TokenExtractor(config, checkpoint_path)
    
                       
    extractor.extract_and_save(train_loader, train_save_path, 'train')
    
                     
    extractor.extract_and_save(val_loader, val_save_path, 'val')
    
    print("\n" + "="*80)
    print("Extraction Complete!")
    print("="*80)
    print(f"Train tokens: {train_save_path}")
    print(f"Val tokens: {val_save_path}")
    
                         
    train_size = os.path.getsize(train_save_path) / (1024**3)
    val_size = os.path.getsize(val_save_path) / (1024**3)
    print(f"\nDisk usage:")
    print(f"  Train: {train_size:.2f} GB")
    print(f"  Val: {val_size:.2f} GB")
    print(f"  Total: {train_size + val_size:.2f} GB")
    print(f"\n💾 Saved as float16 (50% size reduction vs float32)")


if __name__ == '__main__':
    main()
