   
                                               
                                                           
                                                                  
   

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


                                                           

class FeedForward(nn.Module):
                                                                   
    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
       
                                                                           
       
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model {d_model} must be divisible by n_heads {n_heads}"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
                                                        
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        
        self.dropout_p = dropout
        self.proj_dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, L, D = x.shape
        
                                                                              
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
                                         
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )
        
                                                                 
        out = out.transpose(1, 2).reshape(B, L, D)
        
        out = self.proj(out)
        out = self.proj_dropout(out)
        
        return out


                                                          

class MoELayer(nn.Module):
       
                                                     
       
    def __init__(self, d_model: int, num_experts: int = 8, top_k: int = 2, 
                 dropout: float = 0.1, capacity_factor: float = 1.25):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        
                
        self.router = nn.Linear(d_model, num_experts)
        
                 
        self.experts = nn.ModuleList([
            FeedForward(d_model, dropout=dropout) 
            for _ in range(num_experts)
        ])
        
                        
        self.register_buffer('expert_counts', torch.zeros(num_experts))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
           
                             
                                                                         
           
        B, L, D = x.shape
        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)
        
                         
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
                                 
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        output = torch.zeros_like(x)
        
                                                                    
        for i, expert in enumerate(self.experts):
            mask = (top_k_indices == i).any(dim=-1)
            
            if mask.any():
                expert_input = x[mask]
                expert_output = expert(expert_input)
                
                              
                expert_weights = top_k_probs[mask][top_k_indices[mask] == i].unsqueeze(-1)
                output[mask] += expert_output * expert_weights
                
                if self.training:
                    self.expert_counts[i] += mask.sum().item()
        
        aux_loss = self.compute_load_balance_loss(router_probs)
        return output, aux_loss
    
    def compute_load_balance_loss(self, router_probs: torch.Tensor) -> torch.Tensor:
        importance = router_probs.sum(dim=(0, 1))
        top_k_gates, _ = torch.topk(router_probs, self.top_k, dim=-1)
        load = (router_probs >= top_k_gates[:, :, -1:]).float().sum(dim=(0, 1))
        
        mean_importance = importance.mean()
        mean_load = load.mean()
        
        aux_loss = (importance * load).sum() / (mean_importance * mean_load + 1e-6)
        aux_loss = aux_loss / self.num_experts
        return aux_loss


                                                              

class MoETransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(
            config.d_model, config.n_heads, config.attention_dropout
        )
        self.moe = MoELayer(
            config.d_model, config.num_experts, config.top_k,
            config.dropout, config.expert_capacity_factor
        )
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
    
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        moe_out, aux_loss = self.moe(self.norm2(x))
        x = x + moe_out
        return x, aux_loss


class StandardTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(
            config.d_model, config.n_heads, config.attention_dropout
        )
        self.ffn = FeedForward(config.d_model, dropout=config.dropout)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
    
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


                                                      

class PartitionEmbedding(nn.Module):
    def __init__(self, num_partitions: int, d_model: int):
        super().__init__()
        self.embeddings = nn.Parameter(torch.randn(num_partitions, d_model) * 0.02)
    
    def forward(self, partition_ids):
        return self.embeddings[partition_ids]


class PositionalEmbedding(nn.Module):
    def __init__(self, num_patches: int, d_model: int):
        super().__init__()
        position = torch.arange(num_patches).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(num_patches, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, patch_ids):
        return self.pe[patch_ids]


                                                                           

class DecoupledMoEBackbone(nn.Module):
       
                                             
    
                 
                                                              
                                                          
                                          
                                                         
                                                  
       
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
                                          
        self.input_proj = nn.Linear(config.input_dim, config.d_model)
        
                                                            
        self.partition_emb = PartitionEmbedding(config.num_partitions, config.d_model)
        self.pos_emb = PositionalEmbedding(config.num_patches, config.d_model)
        
                                             
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        
                                               
        self.blocks = nn.ModuleList()
        for i in range(config.num_layers):
            if i in config.moe_layer_ids:
                self.blocks.append(MoETransformerBlock(config))
            else:
                self.blocks.append(StandardTransformerBlock(config))
        
                                             
        self.decoder_blocks = nn.ModuleList([
            StandardTransformerBlock(config)
            for _ in range(config.decoder_num_layers)
        ])
        
                                         
        self.pred_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.input_dim)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def gather_visible(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
           
                                                        
        
             
                                                         
                                                          
        
                
                                        
                                                                        
           
        B, L, D = x.shape
        
                                                             
        keep_mask = (mask == 0)           
        
                                       
                                                             
        noise = torch.rand(B, L, device=x.device)                               
        ids_shuffle = torch.argsort(mask + noise, dim=1)           
        ids_restore = torch.argsort(ids_shuffle, dim=1)            
        
                                            
        len_keep = keep_mask.sum(dim=1).min().item()                                                  
        
                               
        ids_keep = ids_shuffle[:, :len_keep]                  
        x_visible = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        
        return x_visible, ids_restore
    
    def pad_and_unshuffle(self, x_visible: torch.Tensor, restore_indices: torch.Tensor,
                          mask_token: torch.Tensor) -> torch.Tensor:
           
                                                             
        
             
                                                         
                                                               
                                                        
        
                
                                                               
           
        B, N_vis, D = x_visible.shape
        L = restore_indices.shape[1]      
        
                                                      
        mask_tokens = mask_token.expand(B, L - N_vis, -1)                    
        
                                             
        x_concat = torch.cat([x_visible, mask_tokens], dim=1)              
        
                                              
        x_full = torch.gather(x_concat, dim=1, 
                             index=restore_indices.unsqueeze(-1).expand(-1, -1, D))
        
        return x_full
    
    def forward(self, tokens: torch.Tensor, partition_ids: torch.Tensor,
                patch_ids: torch.Tensor, mask: torch.Tensor, return_decoding=True, return_encoder_tokens=False) -> Tuple:
           
                              
        
             
                                                   
                                                                  
                                                              
                                                        
        
                
                                                            
                                                      
                                                                           
           
        
                                                                 
                                                 
        x = self.input_proj(tokens)
        x = x + self.partition_emb(partition_ids) + self.pos_emb(patch_ids)
        
                                                                       
        x_visible, restore_indices = self.gather_visible(x, mask)
        
        total_aux_loss = 0.0
        for block in self.blocks:
            if isinstance(block, MoETransformerBlock):
                x_visible, aux_loss = block(x_visible)
                total_aux_loss += aux_loss
            else:
                x_visible = block(x_visible)
        
                                                   
        encoder_tokens = x_visible                       
        
                                               
        global_embedding = x_visible.mean(dim=1)            
    
        if not return_decoding:
            if return_encoder_tokens:
                return None, total_aux_loss, global_embedding, encoder_tokens
            else:
                return None, total_aux_loss, global_embedding
        
                                                                    
                                                             
        x_full = self.pad_and_unshuffle(x_visible, restore_indices, self.mask_token)
        
                                             
                                                                              
        x_full = x_full + self.partition_emb(partition_ids) + self.pos_emb(patch_ids)
        
                                                                         
                                     
        for block in self.decoder_blocks:
            x_full = block(x_full)
        predictions = self.pred_head(x_full)                
                                                                
                                                      
        if return_encoder_tokens:
            return predictions, total_aux_loss, global_embedding, encoder_tokens
        else:
            return predictions, total_aux_loss, global_embedding