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
from scipy.signal import butter, filtfilt

def bandpass_filter(data, lowcut, highcut, fs, order=5):

    # nyq = 0.5 * fs
    # low = lowcut / nyq
    # high = highcut / nyq
    # b, a = butter(order, [low, high], btype='band')
    # y = filtfilt(b, a, data)
    return data

def fold_data():
    """
    按 subject_id 分割数据，确保训练集、验证集、测试集的 subject 互不重叠
    并且测试集固定不变，每次 10 折验证时打乱训练集样本顺序
    """
    length = 1024  # 信号长度
    test_ratio = 0.1  # 测试集占比，固定 10%

    # 设置滤波参数
    fs = 125         # 采样率 (Hz)，根据实际情况调整
    lowcut = 0.5     # PPG 信号低截止频率 (Hz)
    highcut = 8.0    # PPG 信号高截止频率 (Hz)

    # ✅ 读取数据
    fl = h5py.File(os.path.join('data', 'data_with_subjects.hdf5'), 'r')

    # ✅ 收集所有 subject_id
    subject_dict = defaultdict(list)

    for subject_id in tqdm(fl.keys(), desc="Grouping by Subject ID"):
        for ep_id in range(len(fl[subject_id]) // 2):  # 每个 subject_id 下有多个 episodes
            abp = fl[subject_id][f"abp_{ep_id}"][:]
            ppg = fl[subject_id][f"ppg_{ep_id}"][:]
            subject_dict[subject_id].append((ppg, abp))

    subject_ids = np.array(list(subject_dict.keys()))
    np.random.shuffle(subject_ids)  # ✅ 打乱 subject_id 顺序

    # ✅ 固定测试集 (10% subjects) 不变
    num_test_subjects = int(len(subject_ids) * test_ratio)
    test_subjects = subject_ids[:num_test_subjects]  # **固定测试集**

    # ✅ 剩余部分用于训练 & 验证
    train_val_subjects = subject_ids[num_test_subjects:]
    num_subjects = len(train_val_subjects)
    fold_size = num_subjects // 10  # 每 fold 分配的 subject_id 数量

    # 计算全局 abp 的最小值和最大值
    all_abp_mins, all_abp_maxs = [], []
    for subject in subject_dict:
        for ppg, abp in subject_dict[subject]:
            all_abp_mins.append(np.min(abp[:length]))
            all_abp_maxs.append(np.max(abp[:length]))
    
    global_abp_min = np.min(all_abp_mins)
    global_abp_max = np.max(all_abp_maxs)

    for fold_id in tqdm(range(10), desc="Folding Data"):
        # 选择当前 fold 作为验证集
        val_subjects = train_val_subjects[fold_id * fold_size: (fold_id + 1) * fold_size]
        # 其余部分作为训练集
        train_subjects = np.setdiff1d(train_val_subjects, val_subjects)

        # 存储数据
        X_train, Y_train, subject_train = [], [], []
        X_val, Y_val, subject_val = [], [], []

        # 处理训练集：对每个 subject 使用该 subject 的最大最小值归一化 ppg（先 bandpass 滤波）和全局 abp 最小值和最大值进行归一化
        for subject in train_subjects:
            subject_ppg_mins, subject_ppg_maxs = [], []
            for ppg, abp in subject_dict[subject]:
                ppg_clip = ppg[:length]
                # 预先滤波
                ppg_filtered = bandpass_filter(ppg_clip, lowcut, highcut, fs, order=3)
                subject_ppg_mins.append(np.min(ppg_filtered))
                subject_ppg_maxs.append(np.max(ppg_filtered))

            subject_ppg_min = np.min(subject_ppg_mins)
            subject_ppg_max = np.max(subject_ppg_maxs)
            
            # 对该 subject 下的每个 episode 进行处理
            for ppg, abp in subject_dict[subject]:
                # 截取前 length 个点并先滤波
                ppg_clip = ppg[:length]
                ppg_filtered = bandpass_filter(ppg_clip, lowcut, highcut, fs, order=3)
                ppg_processed = ppg_filtered.reshape(length, 1)
                abp_processed = abp[:length].reshape(length, 1)
                
                # 使用该 subject 的参数归一化 ppg
                if subject_ppg_max - subject_ppg_min == 0:
                    ppg_norm = ppg_processed
                else:
                    ppg_norm = (ppg_processed - subject_ppg_min) / (subject_ppg_max - subject_ppg_min)
                
                # 使用全局 abp 的最小值和最大值进行归一化
                if global_abp_max - global_abp_min == 0:
                    abp_norm = abp_processed
                else:
                    abp_norm = (abp_processed - global_abp_min) / (global_abp_max - global_abp_min)
                
                X_train.append(ppg_norm)
                Y_train.append(abp_norm)
                subject_train.append(subject)

        # 处理验证集，同样按 subject 独立归一化 ppg（bandpass 滤波） 和全局 abp 归一化
        for subject in val_subjects:
            subject_ppg_mins, subject_ppg_maxs = [], []
            for ppg, abp in subject_dict[subject]:
                ppg_clip = ppg[:length]
                ppg_filtered = bandpass_filter(ppg_clip, lowcut, highcut, fs, order=3)
                subject_ppg_mins.append(np.min(ppg_filtered))
                subject_ppg_maxs.append(np.max(ppg_filtered))

            subject_ppg_min = np.min(subject_ppg_mins)
            subject_ppg_max = np.max(subject_ppg_maxs)

            for ppg, abp in subject_dict[subject]:
                ppg_clip = ppg[:length]
                ppg_filtered = bandpass_filter(ppg_clip, lowcut, highcut, fs, order=3)
                ppg_processed = ppg_filtered.reshape(length, 1)
                abp_processed = abp[:length].reshape(length, 1)
                
                if subject_ppg_max - subject_ppg_min == 0:
                    ppg_norm = ppg_processed
                else:
                    ppg_norm = (ppg_processed - subject_ppg_min) / (subject_ppg_max - subject_ppg_min)
                
                if global_abp_max - global_abp_min == 0:
                    abp_norm = abp_processed
                else:
                    abp_norm = (abp_processed - global_abp_min) / (global_abp_max - global_abp_min)
                
                X_val.append(ppg_norm)
                Y_val.append(abp_norm)
                subject_val.append(subject)

        # 打乱训练集样本顺序
        train_data = list(zip(X_train, Y_train, subject_train))
        np.random.shuffle(train_data)
        X_train, Y_train, subject_train = zip(*train_data)

        # 转换为 numpy 数组
        X_train, Y_train = np.array(X_train), np.array(Y_train)
        X_val, Y_val = np.array(X_val), np.array(Y_val)

        # 保存当前 fold 的训练集和验证集
        pickle.dump({
            'X_train': X_train, 'Y_train': Y_train,
            'subject_train': subject_train
        }, open(os.path.join('data', f'train_subject_normal_global{fold_id}.p'), 'wb'))
        pickle.dump({
            'X_val': X_val, 'Y_val': Y_val,
            'subject_val': subject_val
        }, open(os.path.join('data', f'val_subject_normal_global{fold_id}.p'), 'wb'))

    print("✅ 10 折交叉验证数据拆分完成！")

    X_test, Y_test, subject_test = [], [], []
    test_calibration = {}  # 用于保存每个 subject 的校准参数
    for subject in test_subjects:
        # 取第一个窗口作为校准
        first_ppg, first_abp = subject_dict[subject][0]
        first_ppg_clip = first_ppg[:length]
        first_abp_clip = first_abp[:length]
        # 对 PPG 进行滤波
        first_ppg_filtered = bandpass_filter(first_ppg_clip, lowcut, highcut, fs, order=3)
        cal_ppg_min = np.min(first_ppg_filtered)
        cal_ppg_max = np.max(first_ppg_filtered)
        cal_abp_min = np.min(first_abp_clip)
        cal_abp_max = np.max(first_abp_clip)
        test_calibration[subject] = {
            'ppg_min': cal_ppg_min,
            'ppg_max': cal_ppg_max,
            'abp_min': global_abp_min,
            'abp_max': global_abp_max
        }
        
        for ppg, abp in subject_dict[subject]:
            ppg_clip = ppg[:length]
            ppg_filtered = bandpass_filter(ppg_clip, lowcut, highcut, fs, order=3)
            ppg_processed = ppg_filtered.reshape(length, 1)
            abp_processed = abp[:length].reshape(length, 1)
            
            if cal_ppg_max - cal_ppg_min == 0:
                ppg_norm = ppg_processed
            else:
                ppg_norm = (ppg_processed - cal_ppg_min) / (cal_ppg_max - cal_ppg_min)
            
            # 使用全局 abp 归一化
            if global_abp_max - global_abp_min == 0:
                abp_norm = abp_processed
            else:
                abp_norm = (abp_processed - global_abp_min) / (global_abp_max - global_abp_min)
            
            X_test.append(ppg_norm)
            Y_test.append(abp_norm)
            subject_test.append(subject)

    X_test, Y_test = np.array(X_test), np.array(Y_test)
    pickle.dump({
        'X_test': X_test, 'Y_test': Y_test,
        'subject_test': subject_test
    }, open(os.path.join('data', 'test_subject_normal_global.p'), 'wb'))
    
    # 保存测试集校准参数
    pickle.dump(test_calibration, open(os.path.join('data', 'test_calibration_params_global.p'), 'wb'))

    print("✅ 10 折交叉验证数据拆分完成！测试集固定不变，并保存了校准参数")




def main():
    fold_data()  # 运行数据拆分


if __name__ == '__main__':
    main()
