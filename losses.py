   
                                                

                
                                                              
                                                       
   

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class ReconstructionLoss(nn.Module):
       
                                          
                            
       
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, reconstructions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
           
             
                                                                             
                                                              
            
                
                                                                          
           
        losses = {}
        total_loss = 0.0
        
        for modality in reconstructions.keys():
            recon = reconstructions[modality]
            target = targets[modality]
            
                              
            loss = self.mse(recon, target)
            losses[f'recon_{modality}'] = loss
            total_loss += loss
        
        losses['recon_total'] = total_loss / len(reconstructions)
        
        return losses


class ContrastiveLoss(nn.Module):
       
                                                       
    
          
                       
                                                                       
                                                                          
                                    
                                                     
                      
                          
       
                                    
       
    def __init__(self, temperature: float = 0.1, num_patches: int = 10):
        super().__init__()
        self.temperature = temperature
        self.num_patches = num_patches
    
    def forward(self, contrastive_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
           
             
                                                                  
                                                                                               
           
                         
        modalities = sorted(list(contrastive_embeddings.keys()))                          
        device = contrastive_embeddings[modalities[0]].device
        
                                                                 
        embeddings_list = [contrastive_embeddings[mod] for mod in modalities]
        embeddings = torch.stack(embeddings_list, dim=0)
        
        M, B_total, D = embeddings.shape
        B = B_total // self.num_patches
        N = self.num_patches
        
                   
        embeddings = F.normalize(embeddings, dim=-1)
        
                                               
                                                                                
        embeddings_flat = embeddings.reshape(M * B_total, D)
        K = M * B_total
        
                                      
                
        sim_matrix = torch.matmul(embeddings_flat, embeddings_flat.t()) / self.temperature
        
                                                      
                                                                            
        
                                                          
                     
        mod_labels = torch.arange(M, device=device).view(-1, 1).repeat(1, B_total).reshape(-1)
        
                                                                               
                     
        sample_labels = torch.arange(B, device=device).view(-1, 1).repeat(1, N).reshape(-1)        
        sample_labels = sample_labels.repeat(M)          
        
                                                                   
                     
        patch_labels = torch.arange(N, device=device).repeat(M * B)
        
                                              
                      
        is_same_mod = mod_labels.unsqueeze(0) == mod_labels.unsqueeze(1)
        is_same_sample = sample_labels.unsqueeze(0) == sample_labels.unsqueeze(1)
        is_same_patch = patch_labels.unsqueeze(0) == patch_labels.unsqueeze(1)
        is_self = torch.eye(K, dtype=torch.bool, device=device)
        
                                 
        
                         
                                                                                  
                                                             
                                                            
        pos_mask = is_same_sample & (~is_same_mod) & is_same_patch
        
                                
                                                   
                   
        neutral_mask = is_same_sample & (~is_same_patch)
        ignore_mask = neutral_mask | is_self
        
                                                          
        
                             
        logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()
        
             
        exp_logits = torch.exp(logits)
        
                                                            
                                                                            
                                      
        denominator_mask = ~ignore_mask
        denominator_sum = (exp_logits * denominator_mask.float()).sum(dim=1, keepdim=True)
        
                         
                                                 
        log_probs = logits - torch.log(denominator_sum + 1e-6)
        
                                       
                                                  
        mean_log_prob_pos = (pos_mask * log_probs).sum(dim=1) / (pos_mask.sum(dim=1).clamp(min=1))
        
                                                                                                            
        has_positives = pos_mask.sum(dim=1) > 0
        loss = -mean_log_prob_pos[has_positives].mean()
        
        return loss


class WhiteningLoss(nn.Module):
       
                                 
                                                                               
       
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def _offdiag_frobenius(C: torch.Tensor) -> torch.Tensor:
                           
        C_offdiag = C - torch.diag_embed(torch.diagonal(C, dim1=-2, dim2=-1))
        return (C_offdiag * C_offdiag).sum(dim=(-2, -1))
    
    def forward(self, proj_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
           
             
                                                                        
           
        losses = []
        
        for mod, X in proj_inputs.items():
            B, N, D = X.shape
                                       
            X_flat = X.reshape(B * N, D).float()
            
                                                  
            X_centered = X_flat - X_flat.mean(dim=0, keepdim=True)
            
                                                
                               
            n_samples = X_flat.shape[0]
            C = (X_centered.T @ X_centered) / max(n_samples - 1, 1)
            loss_cov = self._offdiag_frobenius(C) / (D * (D - 1))
            
                                                                 
                                          
            std = torch.sqrt(X_centered.var(dim=0) + 1e-4)              
                                        
            loss_var = torch.mean(F.relu(1 - std))
            
                                  
            losses.append(loss_cov + loss_var)
        
        return torch.stack(losses).mean()


                              
       
                         
                                                  
       
                                                                        
                          
                                              
                                             
                                                                                     
                                             
    
                                                    
                                           
                                                
                                            
        
                        
                                                                           
        
                                                          
                                                             
                                                                                   
        
                   
                                                               
                                                                              
             
                                                                              
        
                                                                   
                                                            
                                                   
        
                     
                                
                                            
                                    
                          
         
        
                                       

class OrthogonalLoss(nn.Module):
       
                                     
                                         
                                                                                       
       
    def __init__(self):
        super().__init__()

    def forward(self, shared_embeddings: Dict[str, torch.Tensor], 
                specific_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        
        total_loss = torch.tensor(0.0, device=next(iter(shared_embeddings.values())).device)
        num_modalities = 0
        
        for mod in shared_embeddings.keys():
            s = shared_embeddings[mod]              
            t = specific_embeddings[mod]            
            
                                 
            s_flat = s.flatten(0, 1)
            t_flat = t.flatten(0, 1)
            
                                                 
                                  
            cos_sim = F.cosine_similarity(s_flat, t_flat, dim=1)         
            
                             
            loss = torch.mean(cos_sim.pow(2))
            
            total_loss += loss
            num_modalities += 1
            
        return total_loss / num_modalities if num_modalities > 0 else total_loss

class TotalLoss(nn.Module):
    def __init__(self, temperature: float = 0.1, num_patches: int = 10):
        super().__init__()
        self.recon_loss = ReconstructionLoss()
        self.contrastive_loss = ContrastiveLoss(temperature, num_patches=num_patches)
        self.whitening_loss = WhiteningLoss()
                                    
        self.orthogonal_loss = OrthogonalLoss() 
    
    def forward(self, outputs: Dict, targets: Dict, 
                lambda_recon: float = 1.0, 
                lambda_contrastive: float = 0.5,
                lambda_white: float = 0.02,
                lambda_ortho: float = 0.0):                                      
        
                        
        recon_losses = self.recon_loss(outputs['reconstructions'], targets)
        
                     
        contrastive_loss = self.contrastive_loss(outputs['contrastive_embeddings'])
        
                   
        if 'patch_proj_inputs' in outputs and lambda_white > 0:
            whitening_loss = self.whitening_loss(outputs['patch_proj_inputs'])
        else:
            whitening_loss = torch.tensor(0.0, device=contrastive_loss.device)
            
                              
                                                         
        if lambda_ortho > 0 and 'patch_shared_embeddings' in outputs and 'patch_specific_embeddings' in outputs:
            orthogonal_loss = self.orthogonal_loss(outputs['patch_shared_embeddings'], 
                                                   outputs['patch_specific_embeddings'])
        else:
            orthogonal_loss = torch.tensor(0.0, device=contrastive_loss.device)
        
        total_loss = (lambda_recon * recon_losses['recon_total'] + 
                     lambda_contrastive * contrastive_loss +
                     lambda_white * whitening_loss +
                     lambda_ortho * orthogonal_loss)               
        
        loss_dict = {
            'total': total_loss,
            'contrastive': contrastive_loss,
            'white': whitening_loss,
            'ortho': orthogonal_loss,       
            **recon_losses
        }
        
        return total_loss, loss_dict
