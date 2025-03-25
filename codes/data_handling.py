# """
#     Prepares the data splits for 10 fold cross validation
# """

# import h5py
# import numpy as np
# import os
# from tqdm import tqdm
# import pickle


# def fold_data():
#     """
#         folds the data into splits and saves them
#         to perform 10 fold cross validation
#     """

#     length = 1024           # length of the signals

#                                 # we take this starting points of validation data
#                                 # as we have already shuffled the episodes while creating
#                                 # the data.hdf5 file
#     validation_data_start = {
#         0: 90000,
#         1: 0,
#         2: 10000,
#         3: 20000,
#         4: 30000,
#         5: 40000,
#         6: 50000,
#         7: 60000,
#         8: 70000,
#         9: 80000,
#     }

#     for fold_id in tqdm(range(10), desc='Folding Data'):        # iterate for 10 folds

#         fl = h5py.File(os.path.join('data', 'data.hdf5'), 'r')      # load the episode data

#         X_train = []                        # intialize train data
#         Y_train = []

#         X_val = []                          # intialize validation data
#         Y_val = []

#         max_ppg = -10000                    # intialize metadata, min-max of abp,ppg signals
#         min_ppg = 10000
#         max_abp = -10000
#         min_abp = 10000

#         val_start = validation_data_start[fold_id]      # validation data start
#         val_end = val_start + 10000                     # validation data end

#         for i in tqdm(range(0, val_start), desc='Training Data Part 1'):    # training samples before validation samples

#             X_train.append(np.array(fl['data'][i][1][:length]).reshape(length, 1))  # ppg signal
#             Y_train.append(np.array(fl['data'][i][0][:length]).reshape(length, 1))  # abp signal

#             max_ppg = max(max(fl['data'][i][1]), max_ppg)       # update min-max of ppg 
#             min_ppg = min(min(fl['data'][i][1]), min_ppg)

#             max_abp = max(max(fl['data'][i][0]), max_abp)       # update min-max of abp
#             min_abp = min(min(fl['data'][i][0]), min_abp)

        
#         for i in tqdm(range(val_end, 100000), desc='Training Data Part 2'):    # training samples after validation samples

#             X_train.append(np.array(fl['data'][i][1][:length]).reshape(length, 1))  # ppg signal
#             Y_train.append(np.array(fl['data'][i][0][:length]).reshape(length, 1))  # abp signal

#             max_ppg = max(max(fl['data'][i][1]), max_ppg)       # update min-max of ppg 
#             min_ppg = min(min(fl['data'][i][1]), min_ppg)

#             max_abp = max(max(fl['data'][i][0]), max_abp)       # update min-max of abp
#             min_abp = min(min(fl['data'][i][0]), min_abp)

        
#         for i in tqdm(range(val_start, val_end), desc='Validation Data'):

#             X_val.append(np.array(fl['data'][i][1][:length]).reshape(length, 1))  # ppg signal
#             Y_val.append(np.array(fl['data'][i][0][:length]).reshape(length, 1))  # abp signal

#             max_ppg = max(max(fl['data'][i][1]), max_ppg)       # update min-max of ppg 
#             min_ppg = min(min(fl['data'][i][1]), min_ppg)

#             max_abp = max(max(fl['data'][i][0]), max_abp)       # update min-max of abp
#             min_abp = min(min(fl['data'][i][0]), min_abp)


#         fl = None                   # garbage collection


#         X_train = np.array(X_train)             # converting to numpy array
#         X_train -= min_ppg                      # normalizing
#         X_train /= (max_ppg-min_ppg)

#         Y_train = np.array(Y_train)             # converting to numpy array
#         Y_train -= min_abp                      # normalizing
#         Y_train /= (max_abp-min_abp)

#                                                                 # saving the training data split
#         pickle.dump({'X_train': X_train, 'Y_train': Y_train}, open(os.path.join('data', 'train{}.p'.format(fold_id)), 'wb'))

#         X_train = []                   # garbage collection
#         Y_train = []

#         X_val = np.array(X_val)                 # converting to numpy array        
#         X_val -= min_ppg                        # normalizing
#         X_val /= (max_ppg-min_ppg)

#         Y_val = np.array(Y_val)                 # converting to numpy array
#         Y_val -= min_abp                        # normalizing
#         Y_val /= (max_abp-min_abp)

#                                                                 # saving the validation data split
#         pickle.dump({'X_val': X_val, 'Y_val': Y_val}, open(os.path.join('data', 'val{}.p'.format(fold_id)), 'wb'))

#         X_val = []                   # garbage collection
#         Y_val = []
#                                                                 # saving the metadata
#         pickle.dump({'max_ppg': max_ppg,
#                      'min_ppg': min_ppg,
#                      'max_abp': max_abp,
#                      'min_abp': min_abp}, open(os.path.join('data', 'meta{}.p'.format(fold_id)), 'wb'))

#     fl = h5py.File(os.path.join('data', 'data.hdf5'), 'r')      # loading the episode data

#     X_test = []                 # intialize test data
#     Y_test = []

#     for i in tqdm(range(100000, len(fl['data']))):

#         X_test.append(np.array(fl['data'][i][1][:length]).reshape(length, 1))       # ppg signal
#         Y_test.append(np.array(fl['data'][i][0][:length]).reshape(length, 1))       # abp signal

#         # max_ppg = max(max(fl['data'][i][1]), max_ppg)
#         # min_ppg = min(min(fl['data'][i][1]), min_ppg)

#         # max_abp = max(max(fl['data'][i][0]), max_abp)
#         # min_abp = min(min(fl['data'][i][0]), min_abp)

#     X_test = np.array(X_test)           # converting to numpy array
#     X_test -= min_ppg                   # normalizing
#     X_test /= (max_ppg-min_ppg)
    
#     Y_test = np.array(Y_test)           # converting to numpy array
#     Y_test -= min_abp                   # normalizing
#     Y_test /= (max_abp-min_abp)

#                                                                 # saving the test data split
#     pickle.dump({'X_test': X_test,'Y_test': Y_test}, open(os.path.join('data', 'test.p'), 'wb'))


# def main():
    
#     fold_data()         # splits the data for 10 fold cross validation


# if __name__ == '__main__':
#     main()

import h5py
import numpy as np
import os
from tqdm import tqdm
import pickle
from collections import defaultdict
from scipy.signal import butter, filtfilt, find_peaks,iirnotch
import matplotlib.pyplot as plt

# ✅ 1. 基线漂移校正 + 波峰可视化
def baseline_drift_correction(X, plot=False, title=""):
    X = np.array(X).flatten()
    Time_Vector = np.linspace(1, len(X), len(X))

    # **计算峰值**
    peaks, _ = find_peaks(X)
    
    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(X, label="Original Signal")
        plt.plot(peaks, X[peaks], "rx", label="Peaks")
        plt.title(f"Peak Detection: {title}")
        plt.legend()
        plt.savefig('/home/lzqhzo/PPG2ABP/codes/imgs/find_peaks.png')
        plt.close()

    try:
        peak_dist = np.diff(peaks)
        median_peak_dist = int(np.median(peak_dist)) if len(peak_dist) > 0 else 10
        Baseline = np.minimum.accumulate(X)
        poly_order = min(3, median_peak_dist)
        coefs = np.polyfit(Time_Vector, Baseline, poly_order)
        Baseline_Fit = np.polyval(coefs, Time_Vector)
    except:
        coefs = np.polyfit(Time_Vector, X, 3)
        Baseline_Fit = np.polyval(coefs, Time_Vector)

    # **计算基线漂移校正**
    Y = X - Baseline_Fit
    Y -= np.min(Y)
    X_amp, Y_amp = np.ptp(X), np.ptp(Y)
    if Y_amp > 0:
        Y *= (X_amp / Y_amp)
    
    # **绘制基线漂移前后对比**
    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(X, label="Original Signal", linestyle="dashed")
        plt.plot(Y, label="Baseline Corrected Signal", linewidth=2)
        plt.title(f"Baseline Drift Correction: {title}")
        plt.legend()
        plt.savefig('/home/lzqhzo/PPG2ABP/codes/imgs/baseline_drift_correction.png')
        plt.close()

    return Y

# ✅ 2. 计算 PPG 导数（VPG, APG）+ 可视化
def compute_derivatives(PPG_ori, fs=125, lowcut=0.5, highcut=40, order=4, plot=False, title=""):
    nyquist = 0.5 * fs
    low, high = lowcut / nyquist, highcut / nyquist
    b, a = butter(order, [low, high], btype='band')

    def notch_filter(signal, notch_freq=50, fs=125, quality_factor=30):
        nyquist = fs / 2
        w0 = notch_freq / nyquist  # 归一化频率
        b, a = iirnotch(w0, quality_factor)
        return filtfilt(b, a, signal)  # 零相位滤波，避免相位失真

    # 示例：对 PPG 信号进行陷波滤波

    def apply_filter(signal):
        return filtfilt(b, a, signal)

    PPG_ori = np.array(PPG_ori).flatten()
    PPG = notch_filter(PPG_ori, notch_freq=50)
    PPG = notch_filter(PPG, notch_freq=60)
    PPG = apply_filter(PPG)
    # PPG = (PPG - np.mean(PPG)) / np.std(PPG)

    VPG = np.gradient(PPG, 1/fs)
    VPG = apply_filter(VPG)
    VPG = (VPG - np.mean(VPG)) / np.std(VPG)

    APG = np.gradient(VPG, 1/fs)
    APG = apply_filter(APG)
    APG = (APG - np.mean(APG)) / np.std(APG)

    # **绘制 PPG, VPG, APG 对比图**
    if plot:
        plt.figure(figsize=(10, 8))
        plt.subplot(4, 1, 1)
        plt.plot(PPG_ori, label="PPG (original)")
        plt.legend()
        plt.subplot(4, 1, 2)
        plt.plot(PPG, label="PPG (Filtered)", color="b")
        plt.legend()
        plt.subplot(4, 1, 3)
        plt.plot(VPG, label="VPG (First Derivative)", color="g")
        plt.legend()
        plt.subplot(4, 1, 4)
        plt.plot(APG, label="APG (Second Derivative)", color="r")
        plt.legend()
        plt.suptitle(f"PPG Derivative Computation: {title}")
        plt.tight_layout()
        plt.savefig('/home/lzqhzo/PPG2ABP/codes/imgs/compute_derivatives.png')
        plt.close()

    return PPG, VPG, APG


def fold_data():
    """
    按 subject_id 分割数据，确保训练集、验证集、测试集的 subject 互不重叠
    并且测试集固定不变，每次 10 折验证时打乱训练集样本顺序
    """
    length = 1024  # 信号长度
    test_ratio = 0.1  # 测试集占比，固定 10%

    # ✅ 读取数据文件 (不提前加载数据)
    data_path = os.path.join('data', 'data_with_subjects.hdf5')
    fl = h5py.File(data_path, 'r')

    # ✅ 收集所有 subject_id
    subject_ids = np.array(list(fl.keys()))
    np.random.shuffle(subject_ids)  # ✅ 打乱 subject_id 顺序

    # ✅ 固定测试集 (10% subjects) 不变
    num_test_subjects = int(len(subject_ids) * test_ratio)
    test_subjects = subject_ids[:num_test_subjects]  # **固定测试集**
    train_val_subjects = subject_ids[num_test_subjects:]

    num_subjects = len(train_val_subjects)
    fold_size = num_subjects // 10  # 每 fold 分配的 subject_id 数量

    # **计算训练/验证集的 ABP 全局最大值 (减少内存占用)**
    def compute_global_max_abp(subjects):
        max_abp_values = []
        for subject in tqdm(subjects, desc="Computing Global Max ABP"):
            for ep_id in range(len(fl[subject]) // 2):
                abp = fl[subject][f"abp_{ep_id}"][:length]  # 只加载需要的数据
                max_abp_values.append(np.max(abp))
        return max(max_abp_values)

    global_max_abp = compute_global_max_abp(train_val_subjects)

    # **按需加载数据，避免 subject_dict**
    def process_data(subjects, X_list, Y_list, V_list, A_list, subject_list):
        for subject in subjects:
            subject_ppg_mins, subject_ppg_maxs = [], []

            for ep_id in range(len(fl[subject]) // 2):
                ppg = fl[subject][f"ppg_{ep_id}"][:length]
                abp = fl[subject][f"abp_{ep_id}"][:length]
                subject_ppg_mins.append(np.min(ppg))
                subject_ppg_maxs.append(np.max(ppg))

            subject_ppg_min = np.min(subject_ppg_mins)
            subject_ppg_max = np.max(subject_ppg_maxs)

            for ep_id in range(len(fl[subject]) // 2):
                ppg = fl[subject][f"ppg_{ep_id}"][:length]
                abp = fl[subject][f"abp_{ep_id}"][:length]

                # **PPG 归一化 (按 subject 级别)**
                if subject_ppg_max - subject_ppg_min == 0:
                    ppg_norm = ppg
                else:
                    ppg_norm = (ppg - subject_ppg_min) / (subject_ppg_max - subject_ppg_min)

                # **ABP 归一化**
                abp_norm = abp / global_max_abp

                # **计算 PPG 导数**
                ppg_final, vpg, apg = compute_derivatives(ppg_norm)

                X_list.append(ppg_final.reshape(length, 1))
                Y_list.append(abp_norm.reshape(length, 1))
                V_list.append(vpg.reshape(length, 1))
                A_list.append(apg.reshape(length, 1))
                subject_list.append(subject)

    # **交叉验证划分**
    for fold_id in tqdm(range(10), desc="Folding Data"):
        val_subjects = train_val_subjects[fold_id * fold_size: (fold_id + 1) * fold_size]
        train_subjects = np.setdiff1d(train_val_subjects, val_subjects)

        X_train, Y_train, V_train, A_train, subject_train = [], [], [], [], []
        X_val, Y_val, V_val, A_val, subject_val = [], [], [], [], []

        process_data(train_subjects, X_train, Y_train, V_train, A_train, subject_train)
        process_data(val_subjects, X_val, Y_val, V_val, A_val, subject_val)

        # **减少 shuffle 造成的额外内存占用**
        train_data = list(zip(X_train, Y_train, V_train, A_train, subject_train))
        np.random.shuffle(train_data)
        del X_train, Y_train, V_train, A_train, subject_train  # **释放内存**
        X_train, Y_train, V_train, A_train, subject_train = zip(*train_data)

        X_train, Y_train, V_train, A_train = map(np.array, [X_train, Y_train, V_train, A_train])
        X_val, Y_val, V_val, A_val = map(np.array, [X_val, Y_val, V_val, A_val])

        pickle.dump({'X_train': X_train, 'Y_train': Y_train, 'V_train': V_train, 'A_train': A_train, 'subject_train': subject_train},
                    open(os.path.join('data', f'train_subject_preprocess{fold_id}.p'), 'wb'))
        pickle.dump({'X_val': X_val, 'Y_val': Y_val, 'V_val': V_val, 'A_val': A_val, 'subject_val': subject_val},
                    open(os.path.join('data', f'val_subject_preprocess{fold_id}.p'), 'wb'))

        del X_train, Y_train, V_train, A_train, subject_train  # **释放内存**
        del X_val, Y_val, V_val, A_val, subject_val

    print("✅ 10 折交叉验证数据拆分 & 预处理完成！")

    # **处理测试集**
    X_test, Y_test, V_test, A_test, subject_test = [], [], [], [], []
    process_data(test_subjects, X_test, Y_test, V_test, A_test, subject_test)

    pickle.dump({'X_test': X_test, 'Y_test': Y_test, 'V_test': V_test, 'A_test': A_test, 'subject_test': subject_test, 'global_max_abp': global_max_abp},
                open(os.path.join('data', 'test_subject_preprocess.p'), 'wb'))

    print("✅ 测试集预处理完成！")


def main():
    fold_data()  # 运行数据拆分


if __name__ == '__main__':
    main()
