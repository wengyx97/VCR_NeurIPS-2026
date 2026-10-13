import os
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm
import warnings
from multiprocessing import Pool, cpu_count


BASE_DIR = './autodl-tmp/glycemic_wearable_dataset/big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.2'
OUTPUT_H5_PATH = './autodl-fs/bigideas.h5'
WINDOW_SECONDS = 10
MAX_SAMPLES_PER_SUBJECT_DEBUG = None
MIN_DATA_POINTS_RATIO = 0.5
REQUIRE_ALL_MODALITIES = True

MODALITIES = {
    'ppg': 'BVP', 'eda': 'EDA', 'acc': 'ACC', 'temp': 'TEMP'
}
SAMPLING_RATES = {
    'BVP': 64, 'EDA': 4, 'ACC': 32, 'TEMP': 4
}

def is_sample_valid(sample_data_dict):


    if REQUIRE_ALL_MODALITIES:
        for modality_name in MODALITIES.keys():
            if modality_name not in sample_data_dict:
                return False, f"缺少模态 {modality_name}"

            data = sample_data_dict[modality_name]
            if data.size == 0:
                return False, f"模态 {modality_name} 数据为空"


    for modality_name, data in sample_data_dict.items():
        if data.size == 0:
            if REQUIRE_ALL_MODALITIES:
                return False, f"模态 {modality_name} 数据为空"
            else:
                continue


        e4_name = MODALITIES[modality_name]
        expected_points = SAMPLING_RATES[e4_name] * WINDOW_SECONDS


        actual_points = len(data)


        if actual_points < expected_points * MIN_DATA_POINTS_RATIO:
            return False, (f"模态 {modality_name} 数据点不足: "
                          f"{actual_points}/{expected_points} "
                          f"({actual_points/expected_points*100:.1f}%)")

    return True, "valid"

def process_subject(subject_id):


    print(f"[工作进程] 开始处理被试 S{subject_id}...", flush=True)

    subject_path = os.path.join(BASE_DIR, subject_id)
    modal_data = {}

    for new_name, e4_name in MODALITIES.items():
        file_path = os.path.join(subject_path, f'{e4_name}_{subject_id}.csv')
        if not os.path.exists(file_path):
            warnings.warn(f"\n[被试 S{subject_id}] 警告: 缺少文件 {os.path.basename(file_path)}, 跳过该被试。")
            return None

        try:
            df = pd.read_csv(file_path, low_memory=False)
            if df.empty:
                warnings.warn(f"\n[被试 S{subject_id}] 警告: 文件 {os.path.basename(file_path)} 为空。")
                continue

            time_col_name = df.columns[0]
            first_timestamp_str = str(df[time_col_name].iloc[0])

            if '-' in first_timestamp_str and ':' in first_timestamp_str:
                datetimes = pd.to_datetime(df[time_col_name], errors='coerce')
                bad_rows_mask = datetimes.isna()
                if bad_rows_mask.any():
                    df = df[~bad_rows_mask].copy()
                    datetimes = datetimes.dropna()
                if not datetimes.empty:
                    unix_seconds = datetimes.astype(np.int64) // 10**9
                    df['seconds'] = unix_seconds - unix_seconds.iloc[0]
                else:
                    df['seconds'] = np.nan
            else:
                time_deltas = pd.to_timedelta('00:' + df[time_col_name].astype(str), errors='coerce')
                bad_rows_mask = time_deltas.isna()
                if bad_rows_mask.any():
                    df = df[~bad_rows_mask].copy()
                    time_deltas = time_deltas.dropna()
                df['seconds'] = time_deltas.dt.total_seconds()

            df.dropna(subset=['seconds'], inplace=True)
            df = df.set_index('seconds').drop(columns=[time_col_name])

            if new_name != 'acc':
                df.columns = ['value']

            modal_data[new_name] = df
        except Exception as e:
            warnings.warn(f"\n[被试 S{subject_id}] 错误: 处理文件 {os.path.basename(file_path)} 时发生意外错误: {e}。跳过该被试。")
            return None

    if not modal_data:
        warnings.warn(f"\n[被试 S{subject_id}] 警告: 未能成功加载任何模态数据。")
        return None


    for name, df in modal_data.items():
        if df.empty: continue
        for col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            if std > 1e-6:
                modal_data[name][col] = (df[col] - mean) / std
            else:
                modal_data[name][col] = df[col] - mean


    valid_dfs = [df for df in modal_data.values() if not df.empty]
    if not valid_dfs: return None

    start_times = [df.index.min() for df in valid_dfs]
    end_times = [df.index.max() for df in valid_dfs]
    common_start_time = max(start_times)
    common_end_time = min(end_times)

    if common_start_time >= common_end_time: return None

    samples = []
    current_time = common_start_time
    sample_count = 0
    skipped_count = 0
    skip_reasons = {}

    while current_time + WINDOW_SECONDS <= common_end_time:
        window_start = current_time
        window_end = current_time + WINDOW_SECONDS

        sample_data_dict = {}
        for name, df in modal_data.items():
            if df.empty:
                sample_data_dict[name] = np.zeros((0, 3 if name == 'acc' else 1), dtype=np.float32)
                continue
            window_df = df[(df.index >= window_start) & (df.index < window_end)]
            sample_data_dict[name] = window_df.to_numpy(dtype=np.float32)


        is_valid, reason = is_sample_valid(sample_data_dict)

        if is_valid:
            samples.append(sample_data_dict)
            sample_count += 1
        else:
            skipped_count += 1
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

        current_time += WINDOW_SECONDS


        if (sample_count + skipped_count) % 100 == 0:
            print(f"[工作进程] 被试 S{subject_id} 已处理 {sample_count} 个有效样本 (跳过 {skipped_count} 个无效样本)...", flush=True)


        if MAX_SAMPLES_PER_SUBJECT_DEBUG and len(samples) >= MAX_SAMPLES_PER_SUBJECT_DEBUG:
            print(f"[工作进程] 被试 S{subject_id} 已达到 {MAX_SAMPLES_PER_SUBJECT_DEBUG} 个样本上限，提前结束。", flush=True)
            break


    print(f"[工作进程] 被试 S{subject_id} 处理完成:", flush=True)
    print(f"  - 有效样本: {sample_count}", flush=True)
    print(f"  - 跳过样本: {skipped_count}", flush=True)
    if skip_reasons:
        print(f"  - 跳过原因统计:", flush=True)
        for reason, count in sorted(skip_reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"    * {reason}: {count} 次", flush=True)

    return subject_id, samples

def main():
    subject_dirs = sorted([d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d)) and d.isdigit()])
    print(f"找到 {len(subject_dirs)} 个被试进行处理: {subject_dirs}")
    print(f"数据完整性检查配置:")
    print(f"  - 最小数据点比例: {MIN_DATA_POINTS_RATIO*100}%")
    print(f"  - 要求所有模态: {REQUIRE_ALL_MODALITIES}")

    num_processes = 31
    print(f"启动并行处理，使用 {num_processes} 个工作进程...")


    with h5py.File(OUTPUT_H5_PATH, 'w') as hf:
        meta_group = hf.create_group('metadata')
        sr_group = meta_group.create_group('sampling_rates')
        for key, val in MODALITIES.items():
            sr_group.attrs[key] = SAMPLING_RATES[val]

        meta_group.attrs['min_data_points_ratio'] = MIN_DATA_POINTS_RATIO
        meta_group.attrs['require_all_modalities'] = REQUIRE_ALL_MODALITIES
        meta_group.attrs['window_seconds'] = WINDOW_SECONDS


    batch_size = 5
    total_valid_samples = 0
    total_skipped_samples = 0

    for batch_start in range(0, len(subject_dirs), batch_size):
        batch_subjects = subject_dirs[batch_start:batch_start + batch_size]

        with Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap(process_subject, batch_subjects),
                total=len(batch_subjects),
                desc=f"处理批次 {batch_start//batch_size + 1}"
            ))


        with h5py.File(OUTPUT_H5_PATH, 'a') as hf:
            for result in results:
                if result is None: continue
                subject_id, samples = result
                if not samples: continue

                total_valid_samples += len(samples)

                print(f"[主进程] 正在写入被试 S{subject_id} 的 {len(samples)} 个有效样本到HDF5文件...", flush=True)

                subject_group = hf.create_group(f'S{subject_id}')
                for i, sample_dict in enumerate(samples):
                    sample_group = subject_group.create_group(f'sample_{i:05d}')
                    for name, data in sample_dict.items():
                        sample_group.create_dataset(name, data=data, compression='gzip')

                print(f"[主进程] 被试 S{subject_id} 写入完成。", flush=True)

    print(f"\n处理完成！")
    print(f"总计保存 {total_valid_samples} 个有效样本")
    print(f"数据已保存至 {OUTPUT_H5_PATH}")

if __name__ == '__main__':
    main()
