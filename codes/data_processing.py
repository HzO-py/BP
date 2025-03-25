import h5py
import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import torch

# ==========================
# Step 1: 并行处理原始数据
# ==========================
def process_single_file(k):
    """
    处理单个 .mat 文件，并行提取数据
    """
    fs = 125   # 采样率
    t = 10     # PPG 片段长度
    dt = 5     # 片段间隔
    samples_in_episode = round(fs * t)
    d_samples = round(fs * dt)

    f = h5py.File(os.path.join('raw_data', f'Part_{k}.mat'), 'r')
    ky = f'Part_{k}'
    output_dir = 'processed_data'
    os.makedirs(output_dir, exist_ok=True)

    for i in range(len(f[ky])):
        subject_id = f"{k}_{i}"  # 唯一 Subject ID
        signal = np.array(f[f[ky][i][0]][:,0], dtype=np.float32)  # 读取 PPG 信号
        bp = np.array(f[f[ky][i][0]][:,1], dtype=np.float32)  # 读取 ABP 信号
        if len(signal)==0 or len(bp)==0:
            continue

        output_str = '10s,SBP,DBP,Subject_ID\n'
        for j in range(0, len(signal) - samples_in_episode, d_samples):
            sbp = np.max(bp[j:j+samples_in_episode])  
            dbp = np.min(bp[j:j+samples_in_episode])
            output_str += f"{j},{sbp},{dbp},{subject_id}\n"

        with open(os.path.join(output_dir, f'Part_{k}_{i}.csv'), 'w') as fp:
            fp.write(output_str)

def process_data():
    """
    并行处理所有 Part_k.mat 文件
    """
    os.makedirs('processed_data', exist_ok=True)
    num_workers = min(mp.cpu_count(), 4)  
    with mp.Pool(num_workers) as pool:
        pool.map(process_single_file, range(1, 5)) 

# ==========================
# Step 2: 并行下采样
# ==========================
def downsample_data(minThresh=2500, ratio=0.25):
    """
    采用 SBP 和 DBP 直方图进行下采样，避免过采样高频段。
    """
    files = next(os.walk('processed_data'))[2]  # 获取所有 csv 文件

    sbps_dict, dbps_dict = {}, {}  # 存储 SBP 和 DBP 的数据索引
    sbps_cnt, dbps_cnt = {}, {}  # 存储 SBP 和 DBP 的出现次数

    candidates = []  # 记录最终选择的 episodes

    # ✅ 1. 读取所有数据，并按 SBP/DBP 进行分类
    for fl in tqdm(files, desc="读取数据文件"):
        lines = open(os.path.join('processed_data', fl), 'r').read().split('\n')[1:-1]
        for line in lines:
            values = line.split(',')
            file_no = int(fl.split('_')[1])  # 文件编号
            record_no = int(fl.split('.')[0].split('_')[2])  # 记录编号
            episode_st = int(values[0])  # 片段起始点
            sbp = int(float(values[1]))  # SBP
            dbp = int(float(values[2]))  # DBP

            if sbp not in sbps_dict:
                sbps_dict[sbp] = []
                sbps_cnt[sbp] = 0
            sbps_dict[sbp].append((file_no, record_no, episode_st))
            sbps_cnt[sbp] += 1

            if dbp not in dbps_dict:
                dbps_dict[dbp] = []
                dbps_cnt[dbp] = 0
            dbps_dict[dbp].append((file_no, record_no, episode_st, sbp))
            dbps_cnt[dbp] += 1

    # ✅ 2. 按 DBP 进行下采样
    dbp_keys = sorted(dbps_dict.keys())  # 获取所有 DBP 值，并排序
    dbps_taken = {}  # 记录已选择的 DBP 数据
    sbps_taken = {}  # 记录已选择的 SBP 数据

    for dbp in tqdm(dbp_keys, desc="按 DBP 选择数据"):
        cnt = min(int(dbps_cnt[dbp] * ratio), minThresh)  # 计算需要选取的数量
        if cnt == 0:
            continue  # 避免 0 个数据的情况

        indices = np.random.choice(len(dbps_dict[dbp]), size=cnt, replace=False)  # 选择索引
        for ind in indices:
            file_no, record_no, episode_st, sbp = dbps_dict[dbp][ind]
            candidates.append((file_no, record_no, episode_st))
            dbps_taken[dbp] = dbps_taken.get(dbp, 0) + 1  # 记录已选数量
            sbps_taken[sbp] = sbps_taken.get(sbp, 0) + 1

    # ✅ 3. 按 SBP 进行额外的下采样（避免 DBP 选中的数据被重复选择）
    sbp_keys = sorted(sbps_dict.keys())  # 获取所有 SBP 值，并排序

    for sbp in tqdm(sbp_keys, desc="按 SBP 选择数据"):
        cnt = min(int(sbps_cnt[sbp] * ratio), minThresh)  # 计算需要选取的数量
        cnt -= sbps_taken.get(sbp, 0)  # 减去已从 DBP 采样的数量
        cnt = max(cnt, 0)  # 避免负数
        cnt = min(cnt, len(sbps_dict[sbp]))  # 确保不超过可用样本数
        if cnt == 0:
            continue  # 避免 0 个数据的情况

        indices = np.random.choice(len(sbps_dict[sbp]), size=cnt, replace=False)  # 选择索引
        for ind in indices:
            file_no, record_no, episode_st = sbps_dict[sbp][ind]
            candidates.append((file_no, record_no, episode_st))
            sbps_taken[sbp] = sbps_taken.get(sbp, 0) + 1  # 记录已选数量

    # ✅ 4. 保存下采样后的数据
    print(f"🎯 共选择 {len(candidates)} 个 episodes")
    pickle.dump(candidates, open(os.path.join('output','candidates.p'), 'wb'))

    # ✅ 5. 绘制直方图，查看下采样分布
    sbp_counts = [sbps_taken.get(sbp, 0) for sbp in sbp_keys]
    dbp_counts = [dbps_taken.get(dbp, 0) for dbp in dbp_keys]

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.bar(sbp_keys, sbp_counts, color='b')
    plt.title('sbp')

    plt.subplot(2, 1, 2)
    plt.bar(dbp_keys, dbp_counts, color='r')
    plt.title('dbp')

    plt.savefig('imgs/downsample.png')


# ==========================
# Step 3: 并行提取 episodes
# ==========================
def extract_single_episode(file_no, record_no, episode_st):
    """
    处理单个 episode 并用 PyTorch 加速计算
    """
    fs = 125
    t = 10
    samples_in_episode = round(fs * t)

    # ✅ 只在当前进程打开 h5py.File，避免跨进程错误
    with h5py.File(f'./raw_data/Part_{file_no}.mat', 'r') as f:
        ky = f'Part_{file_no}'

        # ✅ 读取 PPG 和 ABP 数据
        try:
            ppg = torch.tensor(f[f[ky][record_no][0]][episode_st:episode_st+samples_in_episode, 0], dtype=torch.float32)
            abp = torch.tensor(f[f[ky][record_no][0]][episode_st:episode_st+samples_in_episode, 1], dtype=torch.float32)
        except KeyError:
            print(f"⚠️ 记录 {file_no}_{record_no} 不存在，跳过 episode {episode_st}")
            return

        # ✅ 检查数据完整性
        if len(ppg) < samples_in_episode or len(abp) < samples_in_episode:
            print(f"⚠️ 跳过 {file_no}_{record_no}_{episode_st}, 数据长度不足 {samples_in_episode}!")
            return

    # ✅ 确保存储目录存在
    os.makedirs('ppgs', exist_ok=True)
    os.makedirs('abps', exist_ok=True)

    # ✅ 保存 PPG 和 ABP
    pickle.dump(ppg, open(os.path.join('ppgs', f'{file_no}_{record_no}_{episode_st}.p'), 'wb'))
    pickle.dump(abp, open(os.path.join('abps', f'{file_no}_{record_no}_{episode_st}.p'), 'wb'))



def extract_episodes():
    """
    并行提取 episodes
    """
    os.makedirs('ppgs', exist_ok=True)
    os.makedirs('abps', exist_ok=True)

    # ✅ 加载下采样后的 candidates
    downsampled_candidates = pickle.load(open('candidates.p', 'rb'))

    # ✅ 采用 starmap 传多个参数
    num_workers = min(mp.cpu_count(), 4)
    with mp.Pool(num_workers) as pool:
        pool.starmap(extract_single_episode, downsampled_candidates)

    print(f"✅ 提取完成，保存 {len(downsampled_candidates)} 条 episodes")

# ==========================
# Step 4: 合并 episodes
# ==========================
def process_episode(file_info):
    """
    读取单个 episode 并返回 subject_id, abp, ppg
    """
    fl = file_info
    subject_id = "_".join(fl.split("_")[:2])  # ✅ 解析 `{file_no}_{record_no}`

    # 读取数据
    abp = pickle.load(open(os.path.join('abps', fl), 'rb'))
    ppg = pickle.load(open(os.path.join('ppgs', fl), 'rb'))

    return subject_id, np.array(abp, dtype=np.float32), np.array(ppg, dtype=np.float32)


def merge_episodes():
    """
    并行合并 episodes 并保存为 HDF5 文件
    """
    os.makedirs('data', exist_ok=True)

    abp_files = sorted(next(os.walk('abps'))[2])  # 读取 ABP 文件
    num_workers = min(mp.cpu_count(), 8)  # ✅ 限制最大进程数，防止 IO 过载

    # ✅ 读取所有数据并行处理
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(process_episode, abp_files, chunksize=10), 
                            total=len(abp_files), desc="Processing Episodes"))

    # ✅ 使用 HDF5 存储数据
    with h5py.File(os.path.join('data', 'data_with_subjects.hdf5'), 'w', libver='latest', swmr=True) as f:
        subject_groups = {}  # 存储已创建的 subject 组

        for subject_id, abp, ppg in tqdm(results, desc="Writing to HDF5"):
            if subject_id not in subject_groups:
                subject_groups[subject_id] = f.create_group(subject_id)  # ✅ 按 `{file_no}_{record_no}` 组织

            # ✅ 计算当前 subject_id 的 episode 数量
            ep_id = str(len(subject_groups[subject_id].keys()) // 2)  # 两个 dataset 代表一个 episode
            subject_groups[subject_id].create_dataset(f"abp_{ep_id}", data=abp)
            subject_groups[subject_id].create_dataset(f"ppg_{ep_id}", data=ppg)

    print(f"✅ 并行合并完成，保存 {len(abp_files)} 条 episodes 到 'data_with_subjects.hdf5' 🚀")


# ==========================
# Step 5: 运行主流程
# ==========================
def main():
    # process_data()  # 并行数据处理
    # downsample_data()  # 并行下采样
    # extract_episodes()  # 并行提取
    merge_episodes()  # 存储最终数据

if __name__ == '__main__':
    main()
