   
                                       
                                                           
   

class BackboneConfig:
                                                              
    
                                                                 
    device = 'cuda'
    num_workers = 24
    pin_memory = True
    
                                                          
                                                                
    train_tokens_path = 'tmp/backbone_tokens/train_tokens.pt'
    val_tokens_path = 'tmp/backbone_tokens/val_tokens.pt'
    
                                                                  
                          
    input_dim = 256  
    d_model = 512                       
    n_heads = 8      
    head_dim = d_model // n_heads
    
                    
    num_layers = 6
    dropout = 0.1
    attention_dropout = 0.1
    
                           
    decoder_num_layers = 2
    decoder_d_model = 512                                            
    
                                              
    moe_layer_ids = [2]                         
    num_experts = 8
    top_k = 2
    expert_capacity_factor = 1.25
    
                            
    num_modalities = 4                       
    num_patches = 10                
    num_token_types = 2           
    seq_length = num_modalities * num_patches * num_token_types      
    
                                                                 
    num_partitions = num_modalities * num_token_types     
    
                                                                  
                    
    random_mask_ratio = 0.75
    
                                            
    temporal_mask_duration = 5
    
                      
    modality_other_token_mask_prob = 0.5
    modality_drop_prob = 0.5                   
    
                              
                                          
    mask_strategy_probs = {
        'random': 0.4,                                         
        'temporal': 0.0,
        'single_modality': 0.4,                      
        'multi_modality': 0.2                              
    }
    
                                                                        
    batch_size = 1024
    max_epochs = 80
    
               
    optimizer = 'AdamW'
    learning_rate = 1e-4
    weight_decay = 0.05
    betas = (0.9, 0.999)
    
                            
    warmup_epochs = 10
    lr_warmup_start = 1e-6
    lr_min = 1e-6
    
                                                            
    lambda_recon = 1.0
    lambda_aux = 1.0
    
                                                                       
    log_interval = 50
    val_interval = 1
    save_interval = 5
    
    checkpoint_dir = 'backbone_checkpoints'
    log_dir = '.backbone_logs'
    
                    
    early_stopping_patience = 15
    
                                                               
    use_amp = True
    grad_clip = 1.0
    
                                                               
    seed = 42