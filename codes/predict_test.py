"""
    Computes the outputs for test data
"""

from torch.utils.data import DataLoader, TensorDataset
from helper_functions import *
from models import MLPRegression, UNetDS64, MultiResUNet1D
import os
import torch
import pickle


# def predict_test_data():
#     """
#         Computes the outputs for test data
#         and saves them in order to avoid recomputing
#     """

#     length = 1024               # length of signal

#     dt = pickle.load(open(os.path.join('data','test.p'),'rb'))      # loading test data
#     X_test = dt['X_test']
#     Y_test = dt['Y_test']   


#     mdl1 = UNetDS64(length)                                             # creating approximation network
#     mdl1.load_weights(os.path.join('models','ApproximateNetwork.h5'))   # loading weights
    
#     Y_test_pred_approximate = mdl1.predict(X_test,verbose=1)            # predicting approximate abp waveform

#     pickle.dump(Y_test_pred_approximate,open('test_output_approximate.p','wb')) # saving the approxmiate predictions


#     mdl2 = MultiResUNet1D(length)                                       # creating refinement network
#     mdl2.load_weights(os.path.join('models','RefinementNetwork.h5'))    # loading weights

#     Y_test_pred = mdl2.predict(Y_test_pred_approximate[0],verbose=1)    # predicting abp waveform

#     pickle.dump(Y_test_pred,open('test_output.p','wb'))                 # saving the predicted abp waeforms

def create_meta_subject_normal():
    # 1) 加载测试集数据 (包含 subject_test)
    dt_test = pickle.load(open(os.path.join('data', 'test_subject_normal.p'), 'rb'))
    X_test = dt_test['X_test']  # (N, length, 1) or similar
    Y_test = dt_test['Y_test']  # (N, length, 1) or similar
    subject_test = dt_test['subject_test']  # 长度为 N 的列表，每个元素是一个 subject_id

    # 2) 加载 subject-level 校准参数
    test_calibration = pickle.load(open(os.path.join('data', 'test_calibration_params.p'), 'rb'))
    # test_calibration[subject] = {
    #   'ppg_min': ...,
    #   'ppg_max': ...,
    #   'abp_min': ...,
    #   'abp_max': ...
    # }

    N = len(subject_test)
    # 创建四个数组，用来存储每条样本对应的 min / max
    min_ppg_arr = np.zeros(N, dtype=np.float32)
    max_ppg_arr = np.zeros(N, dtype=np.float32)
    min_abp_arr = np.zeros(N, dtype=np.float32)
    max_abp_arr = np.zeros(N, dtype=np.float32)

    # 3) 遍历每个样本，取 subject_test[i] 找到该 subject 的校准参数
    for i, subj in enumerate(subject_test):
        subj_params = test_calibration[subj]  # 取到 {'ppg_min','ppg_max','abp_min','abp_max'}
        min_ppg_arr[i] = subj_params['ppg_min']
        max_ppg_arr[i] = subj_params['ppg_max']
        min_abp_arr[i] = subj_params['abp_min']
        max_abp_arr[i] = subj_params['abp_max']

    # 4) 保存 meta 数据
    meta_data = {
        'min_ppg': min_ppg_arr,
        'max_ppg': max_ppg_arr,
        'min_abp': min_abp_arr,
        'max_abp': max_abp_arr
    }
    with open(os.path.join('data', 'meta_subject_normal.p'), 'wb') as f:
        pickle.dump(meta_data, f)

    print("✅ meta_subject_normal.p 生成完成！")

def predict_test_data():
    """
    Computes the outputs for test data using 10 cross-validated models
    and saves them separately.
    """

    length = 1024  # Length of signal
    model_dict = {
        'UNetDS64': UNetDS64,
        'MultiResUNet1D': MultiResUNet1D
    }

    mdlName1 = 'UNetDS64'
    mdlName2 = 'MultiResUNet1D'

    # Load test data
    dt = pickle.load(open(os.path.join('data', 'test_subject_normal.p'), 'rb'))
    X_test = dt['X_test']
    Y_test = dt['Y_test']

    # Convert X_test to PyTorch tensor
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1)

    # Create DataLoader for test data
    test_loader = DataLoader(TensorDataset(X_test_tensor), batch_size=256, shuffle=False)

    for foldname in range(5,10):
        print(f"Processing Fold {foldname+1}")

        # Load and apply approximation network
        mdl1 = model_dict[mdlName1](length).to(device)
        mdl1.load_state_dict(torch.load(f'models/{mdlName1}_subject_normal_model1_fold{foldname}.pth', map_location=device))
        mdl1.eval()

        Y_test_pred_approximate = []
        with torch.no_grad():
            for batch in test_loader:
                Y_test_pred_approximate.append(prepareDataDS(mdl1, batch[0]))

        Y_test_pred_approximate = torch.cat(Y_test_pred_approximate).numpy()
        pickle.dump(Y_test_pred_approximate, open(f'test_subject_normal_output_approximate_fold{foldname}.p', 'wb'))

        torch.cuda.empty_cache()
        del mdl1  # Free memory

        # Load and apply refinement network
        mdl2 = model_dict[mdlName2](length).to(device)
        mdl2.load_state_dict(torch.load(f'models/{mdlName2}_subject_normal_model2_fold{foldname}.pth', map_location=device))
        mdl2.eval()

        # Convert approximations to tensor and create DataLoader
        Y_test_pred_approx_tensor = torch.tensor(Y_test_pred_approximate, dtype=torch.float32).permute(0, 2, 1)
        approx_loader = DataLoader(TensorDataset(Y_test_pred_approx_tensor), batch_size=192, shuffle=False)

        Y_test_pred = []
        with torch.no_grad():
            for batch in approx_loader:
                batch_data = batch[0].to(device)
                Y_test_pred.append(mdl2(batch_data).cpu())

        Y_test_pred = torch.cat(Y_test_pred).numpy()
        print(Y_test_pred.shape)
        pickle.dump(Y_test_pred, open(f'test_subject_normal_output_fold{foldname}.p', 'wb'))

        torch.cuda.empty_cache()
        del mdl2  # Free memory

    print("Testing completed for all folds. Predictions saved separately.")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_mlp_test_data():
    """
    Computes the outputs for test data using 10 cross-validated models
    and saves them separately.
    """

    length = 1024  # Length of signal
    model_dict = {
        'UNetDS64': UNetDS64
    }

    mdlName1 = 'UNetDS64'

    # Load test data
    dt = pickle.load(open(os.path.join('data', 'test.p'), 'rb'))
    X_test = dt['X_test']

    # Convert X_test to PyTorch tensor
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1).to(device)

    # Create DataLoader for test data
    test_loader = DataLoader(TensorDataset(X_test_tensor), batch_size=256, shuffle=False)

    for foldname in range(10):
        print(f"Processing Fold {foldname+1}")

        # **加载 UNetDS64 并冻结**
        mdl1 = model_dict[mdlName1](length).to(device)
        mdl1.load_state_dict(torch.load(f'models/{mdlName1}_model1_fold{foldname}.pth', map_location=device))
        mdl1.eval()  # 进入评估模式

        for param in mdl1.parameters():
            param.requires_grad = False  # 冻结 UNet 参数

        # **加载 MLP**
        example_input = torch.randn(1, 1, length).to(device)
        example_output = mdl1(example_input)
        input_dim = example_output['level4'].view(1, -1).size(1)  # 计算 MLP 输入维度

        mlp = MLPRegression(input_dim).to(device)
        mlp.load_state_dict(torch.load(f'models/MLP_model3_fold{foldname}.pth', map_location=device))
        mlp.eval()

        # **进行 MLP 预测**
        Y_test_pred = []

        with torch.no_grad():
            for batch in test_loader:
                X_batch = batch[0].to(device)

                # **提取 UNetDS64 的 level4 特征**
                features = mdl1(X_batch)
                mlp_features = features['level4'].view(features['level4'].size(0), -1)

                # **MLP 预测 SBP & DBP**
                preds = mlp(mlp_features)  # (batch_size, 2)
                Y_test_pred.append(preds.cpu())

        # **保存 MLP 预测结果**
        Y_test_pred = torch.cat(Y_test_pred, dim=0).numpy()  # (num_samples, 2)
        pickle.dump(Y_test_pred, open(f'test_mlp_output_fold{foldname}.p', 'wb'))

        print(f"Saved MLP predictions to test_mlp_output_fold{foldname}.p")





def main():
    create_meta_subject_normal()
    predict_test_data()     # predicts and stores the outputs of test data to avoid recomputing

if __name__ == '__main__':
    main()