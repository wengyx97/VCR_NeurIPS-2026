   
                                          
                                             

                
                                                            
                                                                  
                                                         
   

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

                                                           

class BasicBlock1D(nn.Module):
       
                                              
       
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
                                    
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


                                                               

class PPGEncoder(nn.Module):
       
                                          
                                        
                                         
       
    def __init__(self, d_model=256):
        super().__init__()
        
              
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
                       
        self.layer1 = self._make_layer(64, 64, num_blocks=3, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=4, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=6, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=3, stride=2)
        
              
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(512, d_model)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
                                 
        layers.append(BasicBlock1D(in_channels, out_channels, stride))
                          
        for _ in range(1, num_blocks):
            layers.append(BasicBlock1D(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
                       
        x = self.stem(x)                    
        x = self.layer1(x)                  
        x = self.layer2(x)                  
        x = self.layer3(x)                  
        x = self.layer4(x)                  
        x = self.avgpool(x)                 
        x = x.flatten(1)                 
        x = self.projection(x)           
        return x


class PPGDecoder(nn.Module):
       
                                              
                                                 
                                            
       
    def __init__(self, d_model=256):
        super().__init__()
        
        self.expansion = nn.Linear(d_model, 512)
        
                                                     
        self.upsample1 = nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1)
        self.upsample2 = nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1)
        self.upsample3 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)
        self.upsample4 = nn.ConvTranspose1d(64, 64, kernel_size=4, stride=2, padding=1)
        self.upsample5 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)
        self.final = nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
                     
        x = self.expansion(x)                        
        x = x.unsqueeze(-1)                             
        
        x = self.relu(self.upsample1(x))                
        x = self.relu(self.upsample2(x))                
        x = self.relu(self.upsample3(x))               
        x = self.relu(self.upsample4(x))                
        x = self.relu(self.upsample5(x))                
        x = self.final(x)                              
        
        return x


                                                               

class ACCEncoder(nn.Module):
       
                                          
                                                
                                         
       
    def __init__(self, d_model=256):
        super().__init__()
        
              
        self.stem = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
                       
        self.layer1 = self._make_layer(64, 64, num_blocks=3, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=4, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=6, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=3, stride=2)
        
              
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(512, d_model)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlock1D(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock1D(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
                       
        x = self.stem(x)                    
        x = self.layer1(x)                  
        x = self.layer2(x)                  
        x = self.layer3(x)                  
        x = self.layer4(x)                  
        x = self.avgpool(x)                 
        x = x.flatten(1)                 
        x = self.projection(x)           
        return x


class ACCDecoder(nn.Module):
       
                          
                                                 
                                            
       
    def __init__(self, d_model=256):
        super().__init__()
        
        self.expansion = nn.Linear(d_model, 512)
        
        self.upsample1 = nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1)
        self.upsample2 = nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1)
        self.upsample3 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)
        self.upsample4 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)
        self.final = nn.Conv1d(32, 3, kernel_size=3, stride=1, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
                     
        x = self.expansion(x)                        
        x = x.unsqueeze(-1)                             
        
        x = self.relu(self.upsample1(x))                
        x = self.relu(self.upsample2(x))                
        x = self.relu(self.upsample3(x))               
        x = self.relu(self.upsample4(x))                
        x = F.interpolate(x, size=32, mode='linear', align_corners=False)               
        x = self.final(x)                              
        
        return x


                                                               

class EDAEncoder(nn.Module):
       
                                          
                                      
                                         
       
    def __init__(self, d_model=256):
        super().__init__()
        
        self.conv1 = nn.Conv1d(1, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=2, stride=2, padding=0)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 512, kernel_size=2, stride=2, padding=0)
        self.bn3 = nn.BatchNorm1d(512)
        
        self.projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, d_model)
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
                      
        x = self.relu(self.bn1(self.conv1(x)))               
        x = self.relu(self.bn2(self.conv2(x)))               
        x = self.relu(self.bn3(self.conv3(x)))               
        x = x.flatten(1)                                  
        x = self.projection(x)                            
        return x


class EDADecoder(nn.Module):
       
                          
                                                 
                                           
       
    def __init__(self, d_model=256):
        super().__init__()
        
        self.expansion = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512)
        )
        
        self.upsample1 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2, padding=0)
        self.upsample2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2, padding=0)
        self.final = nn.Conv1d(128, 1, kernel_size=3, stride=1, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
                     
        x = self.expansion(x)                        
        x = x.unsqueeze(-1)                             
        
        x = self.relu(self.upsample1(x))                
        x = self.relu(self.upsample2(x))                
        x = self.final(x)                             
        
        return x


                                                                

class TEMPEncoder(nn.Module):
       
                                                                      
                                      
                                         
       
    def __init__(self, d_model=256):
        super().__init__()
        
        self.conv1 = nn.Conv1d(1, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=2, stride=2, padding=0)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 512, kernel_size=2, stride=2, padding=0)
        self.bn3 = nn.BatchNorm1d(512)
        
        self.projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, d_model)
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
                      
        x = self.relu(self.bn1(self.conv1(x)))               
        x = self.relu(self.bn2(self.conv2(x)))               
        x = self.relu(self.bn3(self.conv3(x)))               
        x = x.flatten(1)                                  
        x = self.projection(x)                            
        return x


class TEMPDecoder(nn.Module):
       
                                                              
                                                 
                                           
       
    def __init__(self, d_model=256):
        super().__init__()
        
        self.expansion = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512)
        )
        
        self.upsample1 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2, padding=0)
        self.upsample2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2, padding=0)
        self.final = nn.Conv1d(128, 1, kernel_size=3, stride=1, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
                     
        x = self.expansion(x)                        
        x = x.unsqueeze(-1)                             
        
        x = self.relu(self.upsample1(x))                
        x = self.relu(self.upsample2(x))                
        x = self.final(x)                             
        
        return x


                                                                      

class OrthoProjector(nn.Module):
       
                                            
    
                                                                                    
                                       
                                                                                
                                                         
    
         
                                    
                                                                     
                                                   
       
    def __init__(self, d_model: int = 256, rank_ratio: float = 0.5, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.r = max(1, int(d_model * rank_ratio))                 
        
                                                                     
        self.W = nn.Parameter(torch.randn(d_model, self.r) / (d_model ** 0.5))
        
                                             
        self.ln = nn.LayerNorm(d_model, eps=eps)
    
    def _get_basis(self) -> torch.Tensor:
           
                                                      
                                                             
           
        Q, _ = torch.linalg.qr(self.W, mode='reduced')             
        return Q
    
    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
           
             
                                                         
            
                
                                                
                                                    
                                                                         
           
                               
        h_white = self.ln(h)                  
        
                               
        B = self._get_basis()                
        
                                          
        u = torch.matmul(h_white, B)            
        S = torch.matmul(u, B.transpose(0, 1))                  
        
                            
        T = h_white - S.detach()                  
        
        return S, T, h_white


class LearnableHardRankProjector(nn.Module):
       
                                               
                                
       
    def __init__(self, d_model: int = 256, init_rank_ratio: float = 0.5, eps: float = 1e-5, fixed_rank=False):
        super().__init__()
        self.d_model = d_model
        self.fixed_rank_mode = fixed_rank 
        
                                              
                                      
        self.pre_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),         
            nn.Linear(d_model, d_model)
        )

                      
        self.W = nn.Parameter(torch.randn(d_model, d_model))
        nn.init.orthogonal_(self.W)
        
                         
        self.k = int(d_model * init_rank_ratio)
        if not self.fixed_rank_mode:
            self.gate_logits = nn.Parameter(torch.zeros(d_model))
            with torch.no_grad():
                self.gate_logits[:self.k] = 0.5   
                self.gate_logits[self.k:] = -0.5
        else:
            self.register_parameter('gate_logits', None)
            
                                                  
        self.ln = nn.LayerNorm(d_model, eps=eps)

    def _get_hard_gates(self):
           
                                       
           
                        
        soft_gates = torch.sigmoid(self.gate_logits)
        
                        
        hard_gates = (soft_gates > 0.5).float()
        
                                         
        return hard_gates - soft_gates.detach() + soft_gates

    def get_effective_rank(self):
                          
        if self.fixed_rank_mode:
            return self.k
        with torch.no_grad():
            return (self.gate_logits > 0).sum().item()

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                                        
                                            
                               
        h_rectified = self.pre_mlp(h) + h

        magnitude = torch.norm(h_rectified, p=2, dim=-1, keepdim=True) + 1e-6

                                                  
        
                                                        
        h_white = self.ln(h_rectified)
        
                                  
        Q, _ = torch.linalg.qr(self.W)
        
                             
        if self.fixed_rank_mode:
            B = Q[:, :self.k] 
            u = torch.matmul(h_white, B)
            S = torch.matmul(u, B.transpose(0, 1))              
        else:
            gates = self._get_hard_gates()
            u = torch.matmul(h_white, Q)
            u_shared = u * gates 
            S = torch.matmul(u_shared, Q.t())
        
                         
                        
                                                 
                                                
        
                                        
                                                   
                    
        
                            
        T = h_white - S.detach() 
        T = T * magnitude
                                                                                                                                 
                                                                     
                                                       
        
        return S, T, h_white


                                                                                           

class MappingHead(nn.Module):
       
                                                
                                                             
       
    def __init__(self, d_model=256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, d_model)
        )
    
    def forward(self, x):
        return self.network(x)


                                                              

class MultiModalTokenizer(nn.Module):
       
                                                                          
    
                         
                                                                      
                                                     
                                                                      
                                                      
                                                     
                                                                    
                                                             
    
                                                            
                                             
                                                     
                                                         
                                                                  
                                       
                                                                      
       
    def __init__(self, d_model=256, init_rank_ratio=0.5):
        super().__init__()
        
        self.d_model = d_model
        
                  
        self.encoders = nn.ModuleDict({
            'ppg': PPGEncoder(d_model),
            'eda': EDAEncoder(d_model),
            'acc': ACCEncoder(d_model),
            'temp': TEMPEncoder(d_model)
        })
        
                                                     
                                                                                
                                          
                                                                        
                                                     
           
        self.projectors = nn.ModuleDict({
            mod: LearnableHardRankProjector(d_model=d_model, init_rank_ratio=init_rank_ratio, fixed_rank=False)
            for mod in ['ppg', 'eda', 'acc', 'temp']
        })
                                                                
        self.decoders = nn.ModuleDict({
            'ppg': PPGDecoder(d_model),
            'eda': EDADecoder(d_model),
            'acc': ACCDecoder(d_model),
            'temp': TEMPDecoder(d_model)
        })

        self.contrastive_heads = nn.ModuleDict({
            mod: nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model)
            ) for mod in ['ppg', 'eda', 'acc', 'temp']
        })
    
    def encode_patches(self, patches: torch.Tensor, modality: str) -> torch.Tensor:
           
                                         
        
             
                                           
                                   
            
                
                                                                       
           
        B, N = patches.shape[:2]
        encoder = self.encoders[modality]
        
                                
        patches_flat = patches.reshape(B * N, *patches.shape[2:])
        
                
        embeddings_flat = encoder(patches_flat)                  
        
                                         
        embeddings = embeddings_flat.reshape(B, N, self.d_model)
        
        return embeddings
    
    def forward(self, batch: Dict[str, torch.Tensor]):
        outputs = {}
        modalities = ['ppg', 'eda', 'acc', 'temp']
        
                                         
        patch_embeddings = {}
        for mod in modalities:
            patch_embeddings[mod] = self.encode_patches(batch[mod], mod)
        
               
        patch_shared_embeddings = {}    
        patch_specific_embeddings = {}  
        patch_proj_inputs = {}          
        contrastive_embeddings = {}          
        current_ranks = {} 

                                              
        for mod in modalities:
            B, N = patch_embeddings[mod].shape[:2]
            
                                            
            embeddings_flat = patch_embeddings[mod].reshape(B * N, self.d_model)
            
                                          
                                
                                      
                               
                                                                                  
            S_ortho_flat, T_final_flat, h_white_flat = self.projectors[mod](embeddings_flat)
                                                
                              
                                                               
            Z_flat = S_ortho_flat
                                        
                                  
            S_final_flat = Z_flat
            
                                          
                                       
                                                    
                                 
                                                                     
                                                   
                                                                
            
                                
            patch_shared_embeddings[mod] = S_final_flat.reshape(B, N, self.d_model)
            patch_specific_embeddings[mod] = T_final_flat.reshape(B, N, self.d_model)
            patch_proj_inputs[mod] = h_white_flat.reshape(B, N, self.d_model)
            
                                        
                                    
            contrastive_embeddings[mod] = S_final_flat

                          
            if self.training:
                 current_ranks[mod] = self.projectors[mod].get_effective_rank()
                                            
        
        
                                                      
        sample_shared_embeddings = {} 
        sample_specific_embeddings = {}
        
        for mod in modalities:
                             
            sample_shared_embeddings[mod] = patch_shared_embeddings[mod].mean(dim=1)
                                   
            sample_specific_embeddings[mod] = patch_specific_embeddings[mod].mean(dim=1)
        
                                              
        reconstructions = {}
        for mod in modalities:
            B, N = patch_specific_embeddings[mod].shape[:2]
            specific_flat = patch_specific_embeddings[mod].reshape(B * N, self.d_model)
            recon_flat = self.decoders[mod](specific_flat)
            reconstructions[mod] = recon_flat.reshape(B, N, *recon_flat.shape[1:])
        
                         
        outputs['patch_embeddings'] = patch_embeddings
        outputs['patch_shared_embeddings'] = patch_shared_embeddings           
        outputs['patch_specific_embeddings'] = patch_specific_embeddings             
        outputs['sample_shared_embeddings'] = sample_shared_embeddings
        outputs['contrastive_embeddings'] = contrastive_embeddings       
        outputs['sample_specific_embeddings'] = sample_specific_embeddings
        outputs['reconstructions'] = reconstructions
        outputs['patch_proj_inputs'] = patch_proj_inputs
        outputs['current_ranks'] = current_ranks

                                                                        
                               
                                                  
                                                                 
                                                                                                    
                                                                                                                        
                                                                        
        return outputs
    
    def extract_repr(self, batch: Dict[str, torch.Tensor], repr_type: str = 'shared'):
           
                                                                    
        
             
                                                                     
                                                                      
            
                
                                                                      
           
        outputs = self.forward(batch)
        
        if repr_type == 'shared':
            return outputs['sample_shared_embeddings']
        elif repr_type == 'specific':
            return outputs['sample_specific_embeddings']
        else:
            raise ValueError(f"Unknown repr_type: {repr_type}. Must be 'shared' or 'specific'.")
