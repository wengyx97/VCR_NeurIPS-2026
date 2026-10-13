import os
import h5py
import pandas as pd
import numpy as np
from pathlib import Path
from shutil import unpack_archive
import tempfile
import shutil
from multiprocessing import Pool, cpu_count
from functools import partial
import time


MAX_SAMPLES_PER_SUBJECT_DEBUG = None
INPUT_DIR = "./autodl-tmp/UE4W"
OUTPUT_FILE = "./autodl-fs/ue4w.h5"
WINDOW_SIZE = 10
N_PROCESSES = min(32, cpu_count())


SAMPLING_RATES = {
    'ppg': 64,
    'eda': 4,
    'acc': 32,
    'temp': 4
}


FILE_MAPPING = {
    'ppg': 'BVP.csv',
    'eda': 'EDA.csv',
    'acc': 'ACC.csv',
    'temp': 'TEMP.csv'
}


def read_e4_csv(filepath, modality):


    df = pd.read_csv(filepath, header=None)


    start_time = df.iloc[0, 0]


    data = df.iloc[1:].values


    if modality == 'acc':

        data = data.astype(np.float32)
    else:

        data = data.astype(np.float32).reshape(-1, 1)

    return data, start_time


def normalize_data(data):


    if data.size == 0:
        return data

    normalized_data = np.zeros_like(data, dtype=np.float32)


    for col_idx in range(data.shape[1]):
        col_data = data[:, col_idx]
        mean = np.mean(col_data)
        std = np.std(col_data)

        if std > 1e-6:
            normalized_data[:, col_idx] = (col_data - mean) / std
        else:
            normalized_data[:, col_idx] = col_data - mean

    return normalized_data


def extract_zip_to_temp(zip_path):


    temp_dir = tempfile.mkdtemp()
    unpack_archive(zip_path, temp_dir, 'zip')
    return temp_dir


def create_samples(data_dict, window_size, sampling_rates):


    normalized_data_dict = {}
    for modality, data in data_dict.items():
        normalized_data_dict[modality] = normalize_data(data)

    samples = []


    min_samples = float('inf')
    for modality, data in normalized_data_dict.items():
        n_points_per_window = int(window_size * sampling_rates[modality])
        n_samples = len(data) // n_points_per_window
        min_samples = min(min_samples, n_samples)


    for i in range(min_samples):
        sample = {}
        for modality, data in normalized_data_dict.items():
            n_points = int(window_size * sampling_rates[modality])
            start_idx = i * n_points
            end_idx = (i + 1) * n_points
            sample[modality] = data[start_idx:end_idx]
        samples.append(sample)

    return samples


def process_zip_file(zip_path):


    zip_name = os.path.basename(zip_path)


    temp_dir = None
    try:
        temp_dir = extract_zip_to_temp(zip_path)


        data_dict = {}
        for modality, filename in FILE_MAPPING.items():
            file_path = os.path.join(temp_dir, filename)
            if os.path.exists(file_path):
                data, start_time = read_e4_csv(file_path, modality)
                data_dict[modality] = data
            else:
                print(f"Warning: {filename} not found in {zip_name}")


        samples = create_samples(data_dict, WINDOW_SIZE, SAMPLING_RATES)

        return (zip_name, samples)

    except Exception as e:
        print(f"Error processing {zip_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return (zip_name, None)

    finally:

        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def create_h5_file(output_file, all_samples, sampling_rates):


    print(f"\nWriting to {output_file}...")
    write_start = time.time()

    with h5py.File(output_file, 'w') as f:

        metadata = f.create_group('metadata')
        sampling_group = metadata.create_group('sampling_rates')


        for modality, rate in sampling_rates.items():
            sampling_group.attrs[modality] = rate


        metadata.attrs['normalization'] = 'z-score (mean=0, std=1)'
        metadata.attrs['window_seconds'] = WINDOW_SIZE


        subject_group = f.create_group('S001')


        total_samples = len(all_samples)
        for i, sample in enumerate(all_samples):
            sample_name = f'sample_{i:05d}'
            sample_group = subject_group.create_group(sample_name)


            for modality in ['ppg', 'eda', 'acc', 'temp']:
                if modality in sample:
                    sample_group.create_dataset(
                        modality,
                        data=sample[modality],
                        dtype=np.float32,
                        compression='gzip'
                    )


            if (i + 1) % 1000 == 0 or (i + 1) == total_samples:
                elapsed = time.time() - write_start
                print(f"  Written {i+1}/{total_samples} samples ({elapsed:.1f}s)",
                      end='\r', flush=True)

        print()

    write_time = time.time() - write_start
    print(f"Writing completed in {write_time:.2f} seconds")


def main():


    print("="*80)
    print("UE4W Dataset to H5 Converter (Multi-processing with Normalization)")
    print("="*80)
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Window size: {WINDOW_SIZE} seconds")
    print(f"Number of processes: {N_PROCESSES}")
    print(f"Normalization: z-score (mean=0, std=1) applied to each modality")
    if MAX_SAMPLES_PER_SUBJECT_DEBUG is not None:
        print(f"DEBUG MODE: Limited to {MAX_SAMPLES_PER_SUBJECT_DEBUG} samples")
    print("="*80)


    zip_files = sorted(Path(INPUT_DIR).glob('*.zip'))
    print(f"\nFound {len(zip_files)} zip files")

    if not zip_files:
        print(f"No zip files found in {INPUT_DIR}")
        return


    zip_paths = [str(p) for p in zip_files]


    start_time = time.time()
    all_samples = []

    print(f"\nProcessing files with {N_PROCESSES} workers...")

    with Pool(processes=N_PROCESSES) as pool:

        results = pool.imap(process_zip_file, zip_paths)


        for i, (zip_name, samples) in enumerate(results, 1):
            if samples is not None:

                if i % 10 == 0 or i == len(zip_paths):
                    print(f"[{i}/{len(zip_paths)}] Processed, current total: {len(all_samples)} samples",
                          end='\r', flush=True)

                all_samples.extend(samples)


                if MAX_SAMPLES_PER_SUBJECT_DEBUG is not None:
                    if len(all_samples) >= MAX_SAMPLES_PER_SUBJECT_DEBUG:
                        print(f"\nReached debug limit of {MAX_SAMPLES_PER_SUBJECT_DEBUG} samples")
                        all_samples = all_samples[:MAX_SAMPLES_PER_SUBJECT_DEBUG]
                        pool.terminate()
                        break
            else:
                print(f"\n[{i}/{len(zip_paths)}] {zip_name}: FAILED")

    print()
    elapsed_time = time.time() - start_time
    print(f"\nProcessing completed in {elapsed_time:.2f} seconds")
    print(f"Total samples collected: {len(all_samples)}")


    if all_samples:
        create_h5_file(OUTPUT_FILE, all_samples, SAMPLING_RATES)

        file_size = os.path.getsize(OUTPUT_FILE) / (1024**2)
        print(f"H5 file size: {file_size:.2f} MB")


        print("\n" + "="*80)
        print("H5 File Structure Verification:")
        print("="*80)
        with h5py.File(OUTPUT_FILE, 'r') as f:
            print(f"Groups: {list(f.keys())}")
            print(f"Metadata groups: {list(f['metadata'].keys())}")
            print(f"Sampling rates: {dict(f['metadata/sampling_rates'].attrs)}")
            print(f"Normalization: {f['metadata'].attrs.get('normalization', 'N/A')}")
            print(f"Number of samples: {len(list(f['S001'].keys()))}")


            if 'sample_00000' in f['S001']:
                print(f"\nFirst sample structure:")
                for modality in f['S001/sample_00000'].keys():
                    shape = f[f'S001/sample_00000/{modality}'].shape
                    dtype = f[f'S001/sample_00000/{modality}'].dtype
                    expected_length = WINDOW_SIZE * SAMPLING_RATES[modality]
                    print(f"  {modality:4s}: shape={shape}, dtype={dtype}, expected={expected_length}")


            if 'sample_00000' in f['S001']:
                print(f"\nFirst sample data statistics (to verify normalization):")
                for modality in ['ppg', 'eda', 'acc', 'temp']:
                    if modality in f['S001/sample_00000']:
                        data = f[f'S001/sample_00000/{modality}'][:]
                        mean = np.mean(data)
                        std = np.std(data)
                        min_val = np.min(data)
                        max_val = np.max(data)
                        print(f"  {modality:4s}: mean={mean:.6f}, std={std:.6f}, "
                              f"min={min_val:.6f}, max={max_val:.6f}")

        print("="*80)
        print("Done!")
        print("="*80)
    else:
        print("No samples collected, H5 file not created")


if __name__ == '__main__':
    main()
