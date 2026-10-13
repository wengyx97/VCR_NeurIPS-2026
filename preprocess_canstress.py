import os
import h5py
import pandas as pd
import numpy as np
from pathlib import Path
from shutil import unpack_archive
import tempfile
import shutil
from multiprocessing import Pool, cpu_count
import time
from collections import defaultdict


MAX_SAMPLES_PER_SUBJECT_DEBUG = None
INPUT_DIR = "./autodl-tmp/canstress/CAN-STRESS"
LOGBOOK_FILE = "logbook.xlsx"
OUTPUT_FILE_CANNABIS_0 = "./autodl-fs/canstress_cannabis_0.h5"
OUTPUT_FILE_CANNABIS_1 = "./autodl-fs/canstress_cannabis_1.h5"
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


def load_logbook(logbook_path):


    df = pd.read_excel(logbook_path)


    session_to_cannabis = {}
    skipped_count = 0

    for idx, row in df.iterrows():

        if pd.isna(row['E4_Sessions']):
            skipped_count += 1
            continue

        try:
            session_id = str(int(row['E4_Sessions']))
            cannabis = int(row['Cannabis'])
            session_to_cannabis[session_id] = cannabis
        except (ValueError, TypeError) as e:
            print(f"Warning: Skipping row {idx} due to invalid data: {e}")
            skipped_count += 1
            continue

    if skipped_count > 0:
        print(f"Skipped {skipped_count} rows with missing/invalid E4_Sessions values")

    return session_to_cannabis


def read_e4_csv(filepath, modality):


    df = pd.read_csv(filepath, header=None)


    start_time = float(df.iloc[0, 0])


    sample_rate = float(df.iloc[1, 0])


    data = df.iloc[2:].values


    if modality == 'acc':


        data = data.astype(np.float32)
    else:

        data = data.astype(np.float32).reshape(-1, 1)

    return data, start_time, sample_rate


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


def extract_session_id(zip_filename):


    session_id = zip_filename.split('__')[-1].replace('.zip', '')
    return session_id


def create_samples(data_dict, window_size):


    normalized_data_dict = {}
    for modality, info in data_dict.items():
        normalized_data = normalize_data(info['data'])
        normalized_data_dict[modality] = {
            'data': normalized_data,
            'rate': info['rate']
        }

    samples = []


    min_samples = float('inf')
    for modality, info in normalized_data_dict.items():
        data = info['data']
        sample_rate = info['rate']
        n_points_per_window = int(window_size * sample_rate)
        n_samples = len(data) // n_points_per_window
        min_samples = min(min_samples, n_samples)


    for i in range(min_samples):
        sample = {}
        for modality, info in normalized_data_dict.items():
            data = info['data']
            sample_rate = info['rate']
            n_points = int(window_size * sample_rate)
            start_idx = i * n_points
            end_idx = (i + 1) * n_points
            sample[modality] = data[start_idx:end_idx]
        samples.append(sample)

    return samples


def process_zip_file(args):


    zip_path, session_to_cannabis = args
    zip_name = os.path.basename(zip_path)


    session_id = extract_session_id(zip_name)


    if session_id not in session_to_cannabis:

        return (session_id, None, None)

    cannabis = session_to_cannabis[session_id]


    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp()
        unpack_archive(zip_path, temp_dir, 'zip')


        data_dict = {}
        for modality, filename in FILE_MAPPING.items():
            file_path = os.path.join(temp_dir, filename)
            if os.path.exists(file_path):
                data, start_time, sample_rate = read_e4_csv(file_path, modality)
                data_dict[modality] = {
                    'data': data,
                    'rate': sample_rate,
                    'start_time': start_time
                }
            else:
                print(f"Warning: {filename} not found in {zip_name}")


        if len(data_dict) != len(FILE_MAPPING):

            return (session_id, cannabis, None)


        samples = create_samples(data_dict, WINDOW_SIZE)

        return (session_id, cannabis, samples)

    except Exception as e:
        print(f"Error processing {zip_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return (session_id, cannabis, None)

    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def create_h5_file(output_file, subject_samples_dict, sampling_rates):


    print(f"  Writing to {output_file}...")
    write_start = time.time()

    with h5py.File(output_file, 'w') as f:

        metadata = f.create_group('metadata')
        sampling_group = metadata.create_group('sampling_rates')


        for modality, rate in sampling_rates.items():
            sampling_group.attrs[modality] = rate


        metadata.attrs['normalization'] = 'z-score (mean=0, std=1)'
        metadata.attrs['window_seconds'] = WINDOW_SIZE

        total_subjects = len(subject_samples_dict)


        for subject_idx, (session_id, samples) in enumerate(sorted(subject_samples_dict.items()), 1):
            subject_name = f'S{subject_idx:03d}'
            subject_group = f.create_group(subject_name)


            subject_group.attrs['session_id'] = session_id


            for sample_idx, sample in enumerate(samples):
                sample_name = f'sample_{sample_idx:05d}'
                sample_group = subject_group.create_group(sample_name)


                for modality in ['ppg', 'eda', 'acc', 'temp']:
                    if modality in sample:
                        sample_group.create_dataset(
                            modality,
                            data=sample[modality],
                            dtype=np.float32,
                            compression='gzip'
                        )


            if subject_idx % 10 == 0 or subject_idx == total_subjects:
                elapsed = time.time() - write_start
                print(f"    Written {subject_idx}/{total_subjects} subjects ({elapsed:.1f}s)",
                      end='\r', flush=True)

        print()

    write_time = time.time() - write_start
    print(f"  Writing completed in {write_time:.2f} seconds")


def main():


    print("="*80)
    print("CanStress Dataset to H5 Converter (Multi-processing with Normalization)")
    print("="*80)
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output files:")
    print(f"  - Cannabis=0: {OUTPUT_FILE_CANNABIS_0}")
    print(f"  - Cannabis=1: {OUTPUT_FILE_CANNABIS_1}")
    print(f"Window size: {WINDOW_SIZE} seconds")
    print(f"Number of processes: {N_PROCESSES}")
    print(f"Normalization: z-score (mean=0, std=1) applied to each modality")
    if MAX_SAMPLES_PER_SUBJECT_DEBUG is not None:
        print(f"DEBUG MODE: Limited to {MAX_SAMPLES_PER_SUBJECT_DEBUG} samples per subject")
    print("="*80)


    logbook_path = os.path.join(INPUT_DIR, LOGBOOK_FILE)
    if not os.path.exists(logbook_path):
        print(f"Error: Logbook file not found: {logbook_path}")
        return

    print(f"\nLoading logbook from {logbook_path}...")
    session_to_cannabis = load_logbook(logbook_path)
    print(f"Loaded {len(session_to_cannabis)} valid session mappings")


    cannabis_counts = defaultdict(int)
    for cannabis in session_to_cannabis.values():
        cannabis_counts[cannabis] += 1
    print(f"Cannabis distribution: Cannabis=0: {cannabis_counts[0]}, Cannabis=1: {cannabis_counts[1]}")


    zip_files = sorted(Path(INPUT_DIR).glob('*.zip'))
    print(f"\nFound {len(zip_files)} zip files")

    if not zip_files:
        print(f"No zip files found in {INPUT_DIR}")
        return


    zip_paths = [(str(p), session_to_cannabis) for p in zip_files]


    start_time = time.time()


    results_by_cannabis = {0: {}, 1: {}}
    skipped_count = 0
    processed_count = 0

    print(f"\nProcessing files with {N_PROCESSES} workers...")

    with Pool(processes=N_PROCESSES) as pool:
        results = pool.imap(process_zip_file, zip_paths)

        for i, (session_id, cannabis, samples) in enumerate(results, 1):
            if samples is not None and cannabis is not None:

                if MAX_SAMPLES_PER_SUBJECT_DEBUG is not None:
                    samples = samples[:MAX_SAMPLES_PER_SUBJECT_DEBUG]

                results_by_cannabis[cannabis][session_id] = samples
                processed_count += 1


                if i % 20 == 0 or i == len(zip_paths):
                    print(f"[{i}/{len(zip_paths)}] Processed: {processed_count}, Skipped: {skipped_count}",
                          end='\r', flush=True)
            else:
                skipped_count += 1

    print()
    elapsed_time = time.time() - start_time
    print(f"\nProcessing completed in {elapsed_time:.2f} seconds")
    print(f"Total processed: {processed_count}, Total skipped: {skipped_count}")


    for cannabis_value in [0, 1]:
        subject_samples = results_by_cannabis[cannabis_value]

        if not subject_samples:
            print(f"\nNo data for Cannabis={cannabis_value}, skipping H5 creation")
            continue

        output_file = OUTPUT_FILE_CANNABIS_0 if cannabis_value == 0 else OUTPUT_FILE_CANNABIS_1

        total_samples = sum(len(samples) for samples in subject_samples.values())
        print(f"\n{'='*80}")
        print(f"Creating H5 file for Cannabis={cannabis_value}")
        print(f"  Subjects: {len(subject_samples)}")
        print(f"  Total samples: {total_samples}")
        print(f"  Output: {output_file}")

        create_h5_file(output_file, subject_samples, SAMPLING_RATES)

        file_size = os.path.getsize(output_file) / (1024**2)
        print(f"  File size: {file_size:.2f} MB")


        print(f"\n  Verifying {output_file}:")
        with h5py.File(output_file, 'r') as f:
            subjects = list(f.keys())
            subjects.remove('metadata')
            print(f"    Subjects: {len(subjects)}")
            print(f"    Sampling rates: {dict(f['metadata/sampling_rates'].attrs)}")
            print(f"    Normalization: {f['metadata'].attrs.get('normalization', 'N/A')}")


            first_subject = subjects[0]
            session_id = f[first_subject].attrs.get('session_id', 'N/A')
            n_samples = len(list(f[first_subject].keys()))
            print(f"    First subject ({first_subject}): session_id={session_id}, samples={n_samples}")


            first_sample = f'{first_subject}/sample_00000'
            if first_sample in f:
                print(f"    First sample data statistics (to verify normalization):")
                for modality in ['ppg', 'eda', 'acc', 'temp']:
                    if modality in f[first_sample]:
                        data = f[f'{first_sample}/{modality}'][:]
                        mean = np.mean(data)
                        std = np.std(data)
                        print(f"      {modality:4s}: mean={mean:.6f}, std={std:.6f}, shape={data.shape}")

    print("\n" + "="*80)
    print("All processing completed!")
    print("="*80)


if __name__ == '__main__':
    main()
