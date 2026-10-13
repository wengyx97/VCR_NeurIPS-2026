   
                                                    

                
                                                 
   

class Config:

    device = 'cuda'
    num_workers = 24
    pin_memory = True
    
                               
    h5_file_paths = [
        './autodl-fs/bigideas.h5',
        './autodl-fs/canstress_cannabis_0.h5',
        './autodl-fs/ue4w.h5',
    ]
    
    sampling_rates = {
        'ppg': 64,          
        'eda': 4,          
        'acc': 32,                       
        'temp': 4          
    }


    
                          
    window_duration = 10           
    patch_duration = 1             
    num_patches = window_duration // patch_duration              
    
                                             
    patch_sizes = {
        'ppg': (1, 64),                          
        'eda': (1, 4),
        'acc': (3, 32),
        'temp': (1, 4)
    }

    input_channels = {
        'ppg': 1,               
        'eda': 1,               
        'acc': 3,                          
        'temp': 1               
    }
    
                    
    total_hours = 3565      
    total_samples = total_hours * 3600 // window_duration           
    train_ratio = 0.9
    val_ratio = 0.1
    
                                                     
    d_model = 256
                     
    init_rank_ratio = 0.9
    modalities = ['ppg', 'eda', 'acc', 'temp']
    
                                                        
    batch_size = 256                                   
    max_epochs = 80                                                               
    early_stopping_patience = 10
    
               
    optimizer = 'AdamW'
    learning_rate = 1e-4
    weight_decay = 1e-5
    
                            
    warmup_epochs = 10    
    lr_warmup_start = 1e-5
    lr_min = 1e-6
    
                  
    lambda_recon = 0.1
    lambda_contrastive_start = 1
    lambda_contrastive_end = 1
    contrastive_warmup_epochs = 20
    lambda_white = 0.05                                                                     
    
                          
    temperature = 0.1
    
                                                       
    log_interval = 50         
    save_interval = 3                                                                           
    checkpoint_dir = 'checkpoints_good_tokenizer'
    log_dir = 'logs_good_tokenizer'
    
                                                          
    val_interval = 1           
    num_val_samples = 1000                                                                                           
    
