   
                                                                  
                                               
   

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
import pickle
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from config import Config
import multiprocessing as mp


class MultiModalH5Dataset(Dataset):
       
                                                      
                                                          
       
    
    def __init__(self, cached_data: Dict[int, Dict], config: Config):
           
             
                                                                 
                                        
           
        self.data_cache = cached_data
        self.config = config
        self.indices = list(cached_data.keys())
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
                                                  
        data_idx = self.indices[idx]
        data = self.data_cache[data_idx]
        
        return {
            'ppg': torch.from_numpy(data['ppg']).float(),
            'eda': torch.from_numpy(data['eda']).float(),
            'acc': torch.from_numpy(data['acc']).float(),
            'temp': torch.from_numpy(data['temp']).float(),
            'sample_id': data['sample_id']
        }


def split_into_patches(signal: np.ndarray, modality: str, config: Config) -> np.ndarray:
       
                                                 
    
         
                                                                  
                               
                                    
        
            
                                               
       
    sr = config.sampling_rates[modality]
    patch_length = int(sr * config.patch_duration)
    num_patches = config.num_patches
    
                         
    signal = signal.T
    num_channels = signal.shape[0]
    
                                               
    signal_reshaped = signal.reshape(num_channels, num_patches, patch_length)
    
                                                 
    patches = signal_reshaped.transpose(1, 0, 2)
    
    return patches


def process_single_file(args: Tuple) -> Tuple[List[Dict], int, int]:
       
                                                               
                                                      
    
         
                                                                    
        
            
                                                                   
       
    file_idx, h5_path, expected_lengths, config = args
    
    valid_samples_data = []
    total_scanned = 0
    invalid_count = 0
    
    print(f"\n[Process {os.getpid()}] Processing file {file_idx}: {Path(h5_path).name}")
    
    try:
        with h5py.File(h5_path, 'r') as f:
                                                    
            subjects = [k for k in f.keys() if k != 'metadata']
            
            for subject_id in tqdm(subjects, desc=f"File {file_idx}", leave=False):
                subject_group = f[subject_id]
                
                for sample_id in subject_group.keys():
                    total_scanned += 1
                    sample_group = subject_group[sample_id]
                    
                                                                                
                    is_valid = True
                    for modality, expected_len in expected_lengths.items():
                        if modality not in sample_group:
                            is_valid = False
                            break
                        
                                                          
                        if sample_group[modality].shape[0] != expected_len:
                            is_valid = False
                            break
                    
                    if not is_valid:
                        invalid_count += 1
                        continue
                    
                                                                                
                    try:
                                                                                 
                        ppg = sample_group['ppg'][:]
                        eda = sample_group['eda'][:]
                        acc = sample_group['acc'][:]
                        temp = sample_group['temp'][:]
                        
                                                                      
                        ppg_patches = split_into_patches(ppg, 'ppg', config)
                        eda_patches = split_into_patches(eda, 'eda', config)
                        acc_patches = split_into_patches(acc, 'acc', config)
                        temp_patches = split_into_patches(temp, 'temp', config)
                        
                                              
                        valid_samples_data.append({
                            'file_idx': file_idx,
                            'sample_id': f'{subject_id}_{sample_id}',
                            'ppg': ppg_patches,
                            'eda': eda_patches,
                            'acc': acc_patches,
                            'temp': temp_patches
                        })
                        
                    except Exception as e:
                        print(f"  Error processing {subject_id}/{sample_id}: {e}")
                        invalid_count += 1
                        continue
    
    except Exception as e:
        print(f"[Process {os.getpid()}] Error opening file {h5_path}: {e}")
        return [], total_scanned, invalid_count
    
    print(f"[Process {os.getpid()}] File {file_idx} done: "
          f"{len(valid_samples_data)} valid, {invalid_count} invalid")
    
    return valid_samples_data, total_scanned, invalid_count


def scan_and_load_parallel(h5_paths: List[str], config: Config,
                           num_processes: Optional[int] = None) -> Dict[int, Dict]:
       
                                                    
                                                          
                                                                 
    
         
                                           
                                    
                                                                             
        
            
                                                          
       
    expected_lengths = {
        'ppg': 640,               
        'eda': 40,               
        'acc': 320,               
        'temp': 40               
    }
    
    print("\n" + "="*80)
    print("Scanning and loading H5 files (OPTIMIZED - Parallel Mode)...")
    print("Data assumed to be already normalized during preprocessing")
    print("="*80)
    
                                     
    args_list = [
        (file_idx, h5_path, expected_lengths, config)
        for file_idx, h5_path in enumerate(h5_paths)
    ]
    
                                                 
    if len(h5_paths) > 1 and num_processes != 1:
        if num_processes is None:
            num_processes = min(len(h5_paths), mp.cpu_count())
        
        print(f"Using {num_processes} processes for parallel loading...\n")
        
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(process_single_file, args_list)
    else:
                                                            
        print("Using single process mode...\n")
        results = [process_single_file(args) for args in args_list]
    
                                        
    all_samples_data = []
    total_scanned = 0
    total_invalid = 0
    
    for samples_data, scanned, invalid in results:
        all_samples_data.extend(samples_data)
        total_scanned += scanned
        total_invalid += invalid
    
                               
    data_cache = {idx: data for idx, data in enumerate(all_samples_data)}
    
                                  
    total_size = sum(
        data['ppg'].nbytes + data['eda'].nbytes + 
        data['acc'].nbytes + data['temp'].nbytes
        for data in data_cache.values()
    )
    
    print(f"\n{'='*80}")
    print(f"Scan & Load Summary:")
    print(f"  Total scanned:    {total_scanned:,}")
    print(f"  Valid samples:    {len(data_cache):,} ({len(data_cache)/total_scanned*100:.2f}%)")
    print(f"  Invalid samples:  {total_invalid:,} ({total_invalid/total_scanned*100:.2f}%)")
    print(f"  Memory usage:     {total_size / 1024**3:.2f} GB")
    print(f"{'='*80}\n")
    
    return data_cache


def scan_and_load_cached(h5_paths: List[str], config: Config,
                         cache_dir: str = './cache',
                         force_reload: bool = False,
                         num_processes: Optional[int] = None) -> Dict[int, Dict]:
       
                                                    
                                                                  
                                                                 
    
         
                                           
                                    
                                                 
                                                              
                                                               
        
            
                                                          
       
                                                   
    cache_key = '_'.join([Path(p).stem for p in h5_paths])
    cache_key += f'_patches{config.num_patches}'
    cache_file = os.path.join(cache_dir, f'processed_data_{cache_key}.pkl')
    
                            
    if not force_reload and os.path.exists(cache_file):
        print(f"\n{'='*80}")
        print(f"Loading processed data from cache...")
        print(f"  Cache file: {cache_file}")
        
        try:
            with open(cache_file, 'rb') as f:
                data_cache = pickle.load(f)
            
                                    
            total_size = sum(
                data['ppg'].nbytes + data['eda'].nbytes + 
                data['acc'].nbytes + data['temp'].nbytes
                for data in data_cache.values()
            )
            
            print(f" Loaded {len(data_cache):,} samples from cache")
            print(f"  Memory usage: {total_size / 1024**3:.2f} GB")
            print(f"{'='*80}\n")
            return data_cache
            
        except Exception as e:
            print(f"  Cache loading failed: {e}")
            print("  Re-scanning and loading...\n")
    
                                                               
    data_cache = scan_and_load_parallel(h5_paths, config, num_processes)
    
                                  
    try:
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Saving processed data to cache...")
        
        with open(cache_file, 'wb') as f:
            pickle.dump(data_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        cache_size = os.path.getsize(cache_file) / 1024**3
        print(f" Cached {len(data_cache):,} samples to {cache_file}")
        print(f"  Cache file size: {cache_size:.2f} GB\n")
        
    except Exception as e:
        print(f"  Warning: Cache saving failed: {e}\n")
    
    return data_cache


def create_dataloaders(config: Config, 
                      use_cache: bool = True,
                      force_reload: bool = False,
                      num_processes: Optional[int] = None) -> Tuple[DataLoader, DataLoader]:
       
                                                               
                                                                 
    
         
                                    
                                                                            
                                                              
                                                                             
        
            
                                           
       
                                                                      
    if use_cache:
        all_data = scan_and_load_cached(
            config.h5_file_paths, 
            config,
            cache_dir='./cache',
            force_reload=force_reload,
            num_processes=num_processes
        )
    else:
        all_data = scan_and_load_parallel(
            config.h5_file_paths,
            config,
            num_processes=num_processes
        )
    
    if len(all_data) == 0:
        raise ValueError("No valid samples found in H5 files!")
    
                                                  
    all_indices = list(all_data.keys())
    
    if len(all_indices) < config.total_samples:
        print(f"  Warning: Only {len(all_indices):,} valid samples available,")
        print(f"    less than requested {config.total_samples:,}")
        print(f"    Using all {len(all_indices):,} available samples\n")
        selected_indices = all_indices
    else:
        random.seed(42)
        selected_indices = random.sample(all_indices, config.total_samples)
        print(f" Selected {len(selected_indices):,} samples from {len(all_indices):,} available\n")
    
                           
    selected_data = {idx: all_data[idx] for idx in selected_indices}
    
                          
    random.seed(42)
    indices = list(selected_data.keys())
    random.shuffle(indices)
    
    split_idx = int(len(indices) * config.train_ratio)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_data = {idx: selected_data[idx] for idx in train_indices}
    val_data = {idx: selected_data[idx] for idx in val_indices}
    
    print(f"Dataset split:")
    print(f"  Train samples: {len(train_data):,} ({len(train_data)/len(selected_data)*100:.1f}%)")
    print(f"  Val samples:   {len(val_data):,} ({len(val_data)/len(selected_data)*100:.1f}%)")
    print()
    
                     
    train_dataset = MultiModalH5Dataset(train_data, config)
    val_dataset = MultiModalH5Dataset(val_data, config)
    
                        
                                                         
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,                                               
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,                          
        pin_memory=True
    )
    
    print(f"DataLoaders created:")
    print(f"  Train batches: {len(train_loader):,}")
    print(f"  Val batches:   {len(val_loader):,}")
    print(f"  Batch size:    {config.batch_size}")
    print("="*80 + "\n")
    
    return train_loader, val_loader