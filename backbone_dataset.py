   
                                 
                                                                                      
                                               
   

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Tuple


class BackboneDataset(Dataset):
       
                                                              

                                                                            

                       
                                              
                                                        
                            
                                                                                              
                                                                               
                                                                                     
                                                     
                                                                                   
                                                                                           
       

    def __init__(self, pt_path: str, config, mode: str = 'train'):
        self.config = config
        self.mode = mode

        print(f"Loading {mode} tokens from {pt_path}...")
        try:
            data = torch.load(pt_path, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"Error loading {pt_path}: {e}")
            raise e

        self.S_tokens = data['S_tokens']                           
        self.T_tokens = data['T_tokens']                           
        del data

        self.num_samples = self.S_tokens.shape[0]

        self.strategies     = list(config.mask_strategy_probs.keys())
        self.strategy_probs = list(config.mask_strategy_probs.values())

        print(f"✓ Loaded {mode} dataset: {self.num_samples} samples")
        print(f"  Memory Mode: float16 storage (converts to float32 on-the-fly)")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        S = self.S_tokens[idx].float()                
        T = self.T_tokens[idx].float()

        tokens       = self._interleave_ST(S, T)
        partition_ids = self._get_partition_ids()
        patch_ids    = self._get_patch_ids()

        if self.mode == 'train':
            strategy = np.random.choice(self.strategies, p=self.strategy_probs)
        else:
            strategy = 'random'

        mask, weight_mask = self._apply_masking(strategy, tokens.shape[0])

        return {
            'tokens'       : tokens,
            'target'       : tokens.clone(),
            'mask'         : mask,
            'weight_mask'  : weight_mask,
            'partition_ids': partition_ids,
            'patch_ids'    : patch_ids
        }

    def _interleave_ST(self, S: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
           
                                                       
                                          
                                                           
                                                                                   
           
        stacked = torch.stack([S, T], dim=2)                         
        tokens  = stacked.permute(1, 0, 2, 3).reshape(-1, S.shape[-1])
        return tokens

    def _get_partition_ids(self) -> torch.Tensor:
        base = torch.arange(self.config.num_partitions)
        return base.repeat(self.config.num_patches)

    def _get_patch_ids(self) -> torch.Tensor:
        return torch.arange(self.config.num_patches).repeat_interleave(self.config.num_partitions)

    def _apply_masking(self, strategy: str, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if strategy == 'random':
            return self._random_mask(seq_len)
        elif strategy == 'temporal':
            return self._temporal_mask(seq_len)
        elif strategy == 'single_modality':                
            return self._single_modality_dropout(seq_len)
        elif strategy == 'multi_modality':                 
            return self._multi_modality_dropout(seq_len)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


    def _single_modality_dropout(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
           
                             
                          
                             
                                                            
           
        mask = torch.zeros(seq_len)
        weight_mask = torch.ones(seq_len)
        
        M = self.config.num_modalities
        K = self.config.num_partitions

                         
        dropped_mod = np.random.randint(M)
        dropped_mods = np.array([dropped_mod])

                         
        pos_arr = np.arange(seq_len)
        mod_ids = (pos_arr % K) // 2
        is_drop = np.isin(mod_ids, dropped_mods)
        is_t    = (pos_arr % 2 == 1)

        drop_t  = torch.from_numpy(is_drop)
        t_tok_t = torch.from_numpy(is_t)

                              
        mask[drop_t] = 1.0

                                                  
        weight_mask[drop_t & t_tok_t] = 0.0

                               
        p = getattr(self.config, "modality_other_token_mask_prob", 0.5)
        other_t   = ~drop_t
        rand_mask = torch.from_numpy(np.random.random(seq_len) < p)
        mask[other_t & rand_mask] = 1.0

        return mask, weight_mask


    def _multi_modality_dropout(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
           
                                  
                                               
                               
                                            
                                       
           
        mask = torch.zeros(seq_len)
        weight_mask = torch.ones(seq_len)
        
        M = self.config.num_modalities
        K = self.config.num_partitions

                                           
        drop_prob = getattr(self.config, "modality_drop_prob", 0.5)
        while True:
            dropped_flags = np.random.random(M) < drop_prob
            num_dropped = int(dropped_flags.sum())
            if num_dropped in [2, 3]:                 
                break
                
        dropped_mods = np.where(dropped_flags)[0]

                         
        pos_arr = np.arange(seq_len)
        mod_ids = (pos_arr % K) // 2
        is_drop = np.isin(mod_ids, dropped_mods)
        
        drop_t = torch.from_numpy(is_drop)

                               
        mask[drop_t] = 1.0

                                      
        weight_mask[drop_t] = 0.0

                               
        p = getattr(self.config, "modality_other_token_mask_prob", 0.5)
        remain_t  = ~drop_t
        rand_mask = torch.from_numpy(np.random.random(seq_len) < p)
        mask[remain_t & rand_mask] = 1.0

        return mask, weight_mask

    def _random_mask(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
                                                
        mask        = torch.rand(seq_len) < self.config.random_mask_ratio
        weight_mask = torch.ones(seq_len)
        return mask.float(), weight_mask

    def _temporal_mask(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
                                                                     
        mask        = torch.zeros(seq_len)
        weight_mask = torch.ones(seq_len)

        start_patch = np.random.randint(
            0, self.config.num_patches - self.config.temporal_mask_duration + 1
        )
        end_patch = start_patch + self.config.temporal_mask_duration

        for patch_id in range(start_patch, end_patch):
            start_idx = patch_id * 8
            mask[start_idx : start_idx + 8] = 1.0

        return mask, weight_mask

    def _modality_dropout(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
           
                                                                           

                                                                            
                                                                                         
                                                                                         

                 
                                                                                        
                                                                           

                                    
                                                              
                                                               
                                                                                        
                                                                                  

                                           
                                                                           
                                                                                      
                                                                                       
           
        mask        = torch.zeros(seq_len)
        weight_mask = torch.ones(seq_len)

        M = self.config.num_modalities       
        P = self.config.num_patches           
        K = self.config.num_partitions       

        assert seq_len == P * K, f"seq_len={seq_len} != P*K={P*K}"

                                                                                
        drop_prob = getattr(self.config, "modality_drop_prob", 0.5)
        while True:
            dropped_flags = np.random.random(M) < drop_prob
            num_dropped   = int(dropped_flags.sum())
            if 1 <= num_dropped <= M - 1:                               
                break

        dropped_mods = np.where(dropped_flags)[0]                       

                                                                                
                                                                    
                                                                        
                                                                                   
        pos_arr  = np.arange(seq_len)
        mod_ids  = (pos_arr % K) // 2
        is_drop  = np.isin(mod_ids, dropped_mods)
        is_t     = (pos_arr % 2 == 1)                                                   

                                                          
        drop_t  = torch.from_numpy(is_drop)
        t_tok_t = torch.from_numpy(is_t)

                                                                                
        mask[drop_t] = 1.0

        p = getattr(self.config, "modality_other_token_mask_prob", 0.5)

        if num_dropped == 1:
                                                                                
                                                 
            weight_mask[drop_t & t_tok_t] = 0.0

                                                                              
            other_t   = ~drop_t
            rand_mask = torch.from_numpy(np.random.random(seq_len) < p)
            mask[other_t & rand_mask] = 1.0

        else:
                                                                                
            weight_mask[drop_t] = 0.0

                                                                                  
            remain_t  = ~drop_t
            rand_mask = torch.from_numpy(np.random.random(seq_len) < p)
            mask[remain_t & rand_mask] = 1.0

        return mask, weight_mask


def create_backbone_dataloaders(config):
                                                 
    train_dataset = BackboneDataset(config.train_tokens_path, config, mode='train')
    val_dataset   = BackboneDataset(config.val_tokens_path,   config, mode='val')

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    return train_loader, val_loader