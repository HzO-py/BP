"""
    Computes the outputs for test data
"""

from torch.utils.data import DataLoader, TensorDataset
from helper_functions import *
from models import *
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
    dt_test = pickle.load(open(os.path.join('data', 'test_subject_normal_global.p'), 'rb'))
    X_test = dt_test['X_test']  # (N, length, 1) or similar
    Y_test = dt_test['Y_test']  # (N, length, 1) or similar
    subject_test = dt_test['subject_test']  # 长度为 N 的列表，每个元素是一个 subject_id

    # 2) 加载 subject-level 校准参数
    test_calibration = pickle.load(open(os.path.join('data', 'test_calibration_params_global.p'), 'rb'))
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
    with open(os.path.join('data', 'meta_subject_normal_global.p'), 'wb') as f:
        pickle.dump(meta_data, f)

    print("✅ meta_subject_normal_global.p 生成完成！")

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
    dt = pickle.load(open(os.path.join('data', 'test_subject_normal_global.p'), 'rb'))
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
        mdl1.load_state_dict(torch.load(f'models/{mdlName1}_subject_normal_global_model1_fold{foldname}.pth', map_location=device))
        mdl1.eval()

        Y_test_pred_approximate = []
        with torch.no_grad():
            for batch in test_loader:
                Y_test_pred_approximate.append(prepareDataDS(mdl1, batch[0]))

        Y_test_pred_approximate = torch.cat(Y_test_pred_approximate).numpy()
        pickle.dump(Y_test_pred_approximate, open(f'output/test_subject_normal_global_output_approximate_fold{foldname}.p', 'wb'))

        torch.cuda.empty_cache()
        del mdl1  # Free memory

        # Load and apply refinement network
        mdl2 = model_dict[mdlName2](length).to(device)
        mdl2.load_state_dict(torch.load(f'models/{mdlName2}_subject_normal_global_model2_fold{foldname}.pth', map_location=device))
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
        pickle.dump(Y_test_pred, open(f'output/test_subject_normal_global_output_fold{foldname}.p', 'wb'))

        torch.cuda.empty_cache()
        del mdl2  # Free memory

    print("Testing completed for all folds. Predictions saved separately.")


def predict_unet_data():
    """
    Computes the outputs for test data using 5 cross-validated models
    and saves them separately.
    """

    length = 1024  # Length of signal
    model_dict = {
        'UNet1d': UNet1d_Model,  # UNet1d model used here
    }

    mdlName = 'UNet1d'  # Specify the model name

    # Load test data
    dt = pickle.load(open(os.path.join('data', 'test_subject_normal.p'), 'rb'))
    X_test = dt['X_test']
    Y_test = dt['Y_test']

    # Convert X_test to PyTorch tensor
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1).to(device)

    # Create DataLoader for test data
    test_loader = DataLoader(TensorDataset(X_test_tensor), batch_size=256, shuffle=False)

    for foldname in range(5,10):  # Adjust to process the desired folds
        print(f"Processing Fold {foldname+1}")

        # Load and apply UNet1d model (trained model)
        model = model_dict[mdlName]().to(device)  # Instantiate the model
        model.load_state_dict(torch.load(f'models/{mdlName}_subject_normal_model1_fold{foldname}.pth', map_location=device))  # Load model weights
        model.eval()  # Set the model to evaluation mode

        Y_test_pred = []  # List to store predictions

        with torch.no_grad():
            for batch in test_loader:
                batch_data = batch[0].to(device)
                Y_test_pred.append(model(batch_data).cpu())  # Get model predictions

        Y_test_pred = torch.cat(Y_test_pred).numpy()  # Concatenate predictions from all batches
        print(f"Predictions shape for fold {foldname}: {Y_test_pred.shape}")

        # Save the predictions for the current fold
        pickle.dump(Y_test_pred, open(f'output/test_subject_normal_unet_fold{foldname}.p', 'wb'))

        torch.cuda.empty_cache()  # Clear GPU memory
        del model  # Free memory by deleting the model

    print("Testing completed for all folds. Predictions saved separately.")

def predict_resnet_data():
    """
    Computes the outputs for test data using 5 cross-validated models
    and saves them separately.
    """

    length = 1024  # Length of signal
    model_dict = {
        'Resnet1d': ResNet1D,  # UNet1d model used here
    }

    mdlName = 'Resnet1d'  # Specify the model name

    # Load test data
    dt = pickle.load(open(os.path.join('data', 'test_subject_normal_global.p'), 'rb'))
    X_test = dt['X_test']
    Y_test = dt['Y_test']

    # Convert X_test to PyTorch tensor
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1).to(device)

    # Create DataLoader for test data
    test_loader = DataLoader(TensorDataset(X_test_tensor), batch_size=256, shuffle=False)

    for foldname in range(5,10):  # Adjust to process the desired folds
        print(f"Processing Fold {foldname+1}")

        # Load and apply UNet1d model (trained model)
        model = model_dict[mdlName]().to(device)  # Instantiate the model
        model.load_state_dict(torch.load(f'models/{mdlName}_subject_normal_global_model1_fold{foldname}.pth', map_location=device))  # Load model weights
        model.eval()  # Set the model to evaluation mode

        Y_test_pred = []  # List to store predictions

        with torch.no_grad():
            for batch in test_loader:
                batch_data = batch[0].to(device)
                Y_test_pred.append(model(batch_data).cpu())  # Get model predictions

        Y_test_pred = torch.cat(Y_test_pred).numpy()  # Concatenate predictions from all batches
        print(f"Predictions shape for fold {foldname}: {Y_test_pred.shape}")

        # Save the predictions for the current fold
        pickle.dump(Y_test_pred, open(f'output/test_subject_normal_global_resnet_fold{foldname}.p', 'wb'))

        torch.cuda.empty_cache()  # Clear GPU memory
        del model  # Free memory by deleting the model

    print("Testing completed for all folds. Predictions saved separately.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





def main():
    # create_meta_subject_normal()
    predict_resnet_data()     # predicts and stores the outputs of test data to avoid recomputing

if __name__ == '__main__':
    main()