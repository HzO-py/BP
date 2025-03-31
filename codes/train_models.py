# """
#     Trains the PPG2ABP model.
#     10 fold cross validation scheme is followed.
#     Once training is completed the best model based
#     on performance on validation data is selected
#     and independent test is performed.
# """

# from helper_functions import *
# from models import *
# import time
# from tqdm import tqdm
# import pickle
# import os
# from keras.optimizers import Adam

# def train_approximate_network():
#     """
#         Trains the approximate network in 10 fold cross validation manner
#     """
    
#     model_dict = {}                                             # all the different models
#     model_dict['UNet'] = UNet
#     model_dict['UNetLite'] = UNetLite
#     model_dict['UNetWide40'] = UNetWide40
#     model_dict['UNetWide48'] = UNetWide48
#     model_dict['UNetDS64'] = UNetDS64
#     model_dict['UNetWide64'] = UNetWide64
#     model_dict['MultiResUNet1D'] = MultiResUNet1D
#     model_dict['MultiResUNetDS'] = MultiResUNetDS


#     mdlName1 = 'UNetDS64'                                       # approximation network
#     mdlName2 = 'MultiResUNet1D'                                 # refinement network
    
#     length = 1024                                               # length of the signal

#     try:                                                        # create directory to save models
#         os.makedirs('models')
#     except:
#         pass

#     try:                                                        # create directory to save training history
#         os.makedirs('History')
#     except:
#         pass

#                                                                     # 10 fold cross validation
#     for foldname in range(10):

#         print('----------------')
#         print('Training Fold {}'.format(foldname+1))
#         print('----------------')
#                                                                                             # loading training data
#         dt = pickle.load(open(os.path.join('data','train{}.p'.format(foldname)),'rb'))
#         X_train = dt['X_train']
#         Y_train = dt['Y_train']
#                                                                                             # loading validation data
#         dt = pickle.load(open(os.path.join('data','val{}.p'.format(foldname)),'rb'))
#         X_val = dt['X_val']
#         Y_val = dt['Y_val']

#                                                                                             # loading metadata
#         dt = pickle.load(open(os.path.join('data','meta{}.p'.format(foldname)),'rb'))
#         max_ppg = dt['max_ppg']
#         min_ppg = dt['min_ppg']
#         max_abp = dt['max_abp']
#         min_abp = dt['min_abp']


#         Y_train = prepareLabel(Y_train)                                         # prepare labels for training deep supervision
        
#         Y_val = prepareLabel(Y_val)                                             # prepare labels for training deep supervision
    

    
#         mdl1 = model_dict[mdlName1](length)             # create approximation network

#                                                                             # loss = mae, with deep supervision weights
#         mdl1.compile(loss='mean_absolute_error',optimizer='adam',metrics=['mean_squared_error'], loss_weights=[1., 0.9, 0.8, 0.7, 0.6])                                                         


#         checkpoint1_ = ModelCheckpoint(os.path.join('models','{}_model1_fold{}.h5'.format(mdlName1,foldname)), verbose=1, monitor='val_out_loss',save_best_only=True, mode='auto')  
#                                                                         # train approximation network for 100 epochs
#         history1 = mdl1.fit(X_train,{'out': Y_train['out'], 'level1': Y_train['level1'], 'level2':Y_train['level2'], 'level3':Y_train['level3'] , 'level4':Y_train['level4']},epochs=100,batch_size=256,validation_data=(X_val,{'out': Y_val['out'], 'level1': Y_val['level1'], 'level2':Y_val['level2'], 'level3':Y_val['level3'] , 'level4':Y_val['level4']}),callbacks=[checkpoint1_],verbose=1)

#         pickle.dump(history1, open('History/{}_model1_fold{}.p'.format(mdlName1,foldname),'wb'))    # save training history


#         mdl1 = None                                             # garbage collection

#         time.sleep(300)                                         # pause execution for a while to free the gpu
    


# def train_refinement_network():
#     """
#         Trains the refinement network in 10 fold cross validation manner
#     """
    
#     model_dict = {}                                             # all the different models
#     model_dict['UNet'] = UNet
#     model_dict['UNetLite'] = UNetLite
#     model_dict['UNetWide40'] = UNetWide40
#     model_dict['UNetWide48'] = UNetWide48
#     model_dict['UNetDS64'] = UNetDS64
#     model_dict['UNetWide64'] = UNetWide64
#     model_dict['MultiResUNet1D'] = MultiResUNet1D
#     model_dict['MultiResUNetDS'] = MultiResUNetDS


#     mdlName1 = 'UNetDS64'                                       # approximation network
#     mdlName2 = 'MultiResUNet1D'                                 # refinement network
    
#     length = 1024                                               # length of the signal

#                                                                     # 10 fold cross validation
#     for foldname in range(10):

#         print('----------------')
#         print('Training Fold {}'.format(foldname+1))
#         print('----------------')
#                                                                                             # loading training data
#         dt = pickle.load(open(os.path.join('data','train{}.p'.format(foldname)),'rb'))
#         X_train = dt['X_train']
#         Y_train = dt['Y_train']
#                                                                                             # loading validation data
#         dt = pickle.load(open(os.path.join('data','val{}.p'.format(foldname)),'rb'))
#         X_val = dt['X_val']
#         Y_val = dt['Y_val']

#                                                                                             # loading metadata
#         dt = pickle.load(open(os.path.join('data','meta{}.p'.format(foldname)),'rb'))
#         max_ppg = dt['max_ppg']
#         min_ppg = dt['min_ppg']
#         max_abp = dt['max_abp']
#         min_abp = dt['min_abp']


#         Y_train = prepareLabel(Y_train)                                         # prepare labels for training deep supervision
        
#         Y_val = prepareLabel(Y_val)                                             # prepare labels for training deep supervision
    
    
#         mdl1 = model_dict[mdlName1](length)                 # load approximation network
#         mdl1.load_weights(os.path.join('models','{}_model1_fold{}.h5'.format(mdlName1,foldname)))   # load weights

#         X_train = prepareDataDS(mdl1, X_train)          # prepare training data for 2nd stage, considering deep supervision
#         X_val = prepareDataDS(mdl1, X_val)              # prepare validation data for 2nd stage, considering deep supervision

#         mdl1 = None                                 # garbage collection

    
#         mdl2 = model_dict[mdlName2](length)            # create refinement network

#                                                                     # loss = mse
#         mdl2.compile(loss='mean_squared_error',optimizer='adam',metrics=['mean_absolute_error'])

#         checkpoint2_ = ModelCheckpoint(os.path.join('models','{}_model2_fold{}.h5'.format(mdlName2,foldname)), verbose=1, monitor='val_loss',save_best_only=True, mode='auto')  

#                                                                 # train refinement network for 100 epochs
#         history2 = mdl2.fit(X_train,Y_train['out'],epochs=100,batch_size=192,validation_data=(X_val,Y_val['out']),callbacks=[checkpoint2_])

#         pickle.dump(history2, open('History/{}_model2_fold{}.p'.format(mdlName2,foldname),'wb'))    # save training history

#         time.sleep(300)                                         # pause execution for a while to free the gpu





# def main():

#     train_approximate_network()             # train the approximate models for 10 fold
#     train_refinement_network()             # train the refinement models for 10 fold

# if __name__ == '__main__':
#     main()
import gc
import os
import time
import pickle
import scipy.signal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models import *
from tqdm import tqdm

from helper_functions import prepareLabel, prepareDataDS

# 选择 GPU 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs, model_path, is_refinement=False):
    best_val_loss = float('inf')
    counter = 0  # 记录连续未提升的轮数
    patience = 15

    # 💡 **支持多 GPU**
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            X_batch = batch[0].to(device)

            if is_refinement:
                # 💡 细化网络只优化 `out`，不涉及 `level1` 等
                Y_batch = batch[1].to(device)
            else:
                # 💡 近似网络优化 `out`, `level1`, `level2`, `level3`, `level4`
                Y_batch = {
                    'out': batch[1].to(device),
                    'level1': batch[2].to(device),
                    'level2': batch[3].to(device),
                    'level3': batch[4].to(device),
                    'level4': batch[5].to(device),
                }

            optimizer.zero_grad()
            outputs = model(X_batch)

            if is_refinement:
                # 💡 细化网络只计算 `out` 的 MSE 损失
                loss = loss_fn(outputs, Y_batch)
            else:
                # 💡 近似网络计算深度监督损失
                loss = sum(w * loss_fn(outputs[k], Y_batch[k]) for k, w in zip(outputs.keys(), [1., 0.9, 0.8, 0.7, 0.6]))
                

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                X_batch = batch[0].to(device)

                if is_refinement:
                    Y_batch = batch[1].to(device)
                else:
                    Y_batch = {
                        'out': batch[1].to(device),
                        'level1': batch[2].to(device),
                        'level2': batch[3].to(device),
                        'level3': batch[4].to(device),
                        'level4': batch[5].to(device),
                    }
                
                outputs = model(X_batch)

                if is_refinement:
                    loss = loss_fn(outputs, Y_batch)
                else:
                    loss = sum(w * loss_fn(outputs[k], Y_batch[k]) for k, w in zip(outputs.keys(), [1., 0.9, 0.8, 0.7, 0.6]))

                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}")

        # ✅ **Early Stopping 逻辑**
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0  # 重新计数
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), model_path)  
            else:
                torch.save(model.state_dict(), model_path)
            print(f"✅ Model saved at epoch {epoch+1}")
        else:
            counter += 1
            print(f"⚠️ Early Stopping Counter: {counter}/{patience}")

        # **如果 `counter` 超过 `patience`，提前停止训练**
        if counter >= patience:
            print("🛑 Early stopping triggered.")
            break  # 提前停止训练


def train_unet_network():
    model_dict = {
        'UNet1d': UNet1d_Model,
    }

    mdlName1 = 'UNet1d'  
    length = 1024

    os.makedirs('models', exist_ok=True)
    os.makedirs('History', exist_ok=True)

    for foldname in range(5,10):
        print(f"Training Fold {foldname+1}")    

        # 加载数据
        dt = pickle.load(open(f'data/train_subject_normal{foldname}.p', 'rb'))
        X_train, Y_train = dt['X_train'], prepareLabel(dt['Y_train'])

        dt = pickle.load(open(f'data/val_subject_normal{foldname}.p', 'rb'))
        X_val, Y_val = dt['X_val'], prepareLabel(dt['Y_val'])

        # 转换为 Tensor
        X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1).to(device)
        X_val = torch.tensor(X_val, dtype=torch.float32).permute(0, 2, 1).to(device)

        # 构建数据集和 DataLoader
        train_dataset = TensorDataset(X_train, Y_train['out'])
        val_dataset = TensorDataset(X_val, Y_val['out'])

        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

        # 实例化模型
        model = model_dict[mdlName1]().to(device)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        model_path = f'models/{mdlName1}_subject_normal_model1_fold{foldname}.pth'
        train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs=100, model_path=model_path,is_refinement=True)

        torch.cuda.empty_cache()  # 释放 GPU 内存


def train_resnet_network():
    model_dict = {
        'Resnet1d': ResNet1D,  # Use Resnet1d as the model
    }

    mdlName1 = 'Resnet1d'  
    length = 1024

    os.makedirs('models', exist_ok=True)
    os.makedirs('History', exist_ok=True)

    for foldname in range(5, 10):
        print(f"Training Fold {foldname+1}")

        # Load data
        dt = pickle.load(open(f'data/train_subject_normal_global{foldname}.p', 'rb'))
        X_train, Y_train = dt['X_train'], prepareLabel(dt['Y_train'])

        dt = pickle.load(open(f'data/val_subject_normal_global{foldname}.p', 'rb'))
        X_val, Y_val = dt['X_val'], prepareLabel(dt['Y_val'])

        # Convert to Tensor
        X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1).to(device)
        X_val = torch.tensor(X_val, dtype=torch.float32).permute(0, 2, 1).to(device)

        # Prepare datasets and DataLoader
        train_dataset = TensorDataset(X_train, Y_train['sbp_dbp'].squeeze(1))  # Use 'sbp' and 'dbp' as separate labels
        val_dataset = TensorDataset(X_val, Y_val['sbp_dbp'].squeeze(1))

        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

        # Instantiate model
        model = model_dict[mdlName1]()  # Pass the param_model to the ResNet model
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        model_path = f'models/{mdlName1}_subject_normal_global_model1_fold{foldname}.pth'
        
        # Train the model
        train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs=100, model_path=model_path, is_refinement=True)

        torch.cuda.empty_cache()  # Release GPU memory


def train_approximate_network():
    model_dict = {
        'UNet': UNet1D,
        'UNetLite': UNetLite,
        'UNetWide40': UNetWide40,
        'UNetWide48': UNetWide48,
        'UNetDS64': UNetDS64,
        'UNetWide64': UNetWide64,
        'MultiResUNet1D': MultiResUNet1D,
        'MultiResUNetDS': MultiResUNetDS
    }

    mdlName1 = 'UNetDS64'  
    length = 1024

    os.makedirs('models', exist_ok=True)
    os.makedirs('History', exist_ok=True)

    for foldname in range(5,10):
        print(f"Training Fold {foldname+1}")

        # 加载数据
        dt = pickle.load(open(f'data/train_subject_normal_global{foldname}.p', 'rb'))
        X_train, Y_train = dt['X_train'], prepareLabel(dt['Y_train'])

        dt = pickle.load(open(f'data/val_subject_normal_global{foldname}.p', 'rb'))
        X_val, Y_val = dt['X_val'], prepareLabel(dt['Y_val'])
        

        # 转换为 Tensor
        X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1).to(device)
        X_val = torch.tensor(X_val, dtype=torch.float32).permute(0, 2, 1).to(device)


        # **🔥 解决 BUG: 分开传入 dict 内的 Tensors**
        train_dataset = TensorDataset(X_train, Y_train['out'], Y_train['level1'], Y_train['level2'], Y_train['level3'], Y_train['level4'])
        val_dataset = TensorDataset(X_val, Y_val['out'], Y_val['level1'], Y_val['level2'], Y_val['level3'], Y_val['level4'])


        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

        model = model_dict[mdlName1](length).to(device)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        model_path = f'models/{mdlName1}_subject_normal_global_model1_fold{foldname}.pth'
        train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs=50, model_path=model_path)

        torch.cuda.empty_cache()  # 释放 GPU 内存



def train_refinement_network():
    model_dict = {
        'UNet': UNet1D,
        'UNetLite': UNetLite,
        'UNetWide40': UNetWide40,
        'UNetWide48': UNetWide48,
        'UNetDS64': UNetDS64,
        'UNetWide64': UNetWide64,
        'MultiResUNet1D': MultiResUNet1D,
        'MultiResUNetDS': MultiResUNetDS
    }

    mdlName1 = 'UNetDS64'  
    mdlName2 = 'MultiResUNet1D'
    length = 1024

    for foldname in range(5,10):
        print(f"Training Fold {foldname+1}")

        dt = pickle.load(open(f'data/train_subject_normal_global{foldname}.p', 'rb'))
        X_train, Y_train = dt['X_train'], prepareLabel(dt['Y_train'])

        dt = pickle.load(open(f'data/val_subject_normal_global{foldname}.p', 'rb'))
        X_val, Y_val = dt['X_val'], prepareLabel(dt['Y_val'])

        print('prepareLabel done')

        mdl1 = model_dict[mdlName1](length).to(device)
        mdl1.load_state_dict(torch.load(f'models/{mdlName1}_subject_normal_global_model1_fold{foldname}.pth', weights_only=True))
        mdl1.eval()

        X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
        X_val = torch.tensor(X_val, dtype=torch.float32).permute(0, 2, 1)
        train_loader = DataLoader(TensorDataset(X_train), batch_size=256, shuffle=False)
        val_loader = DataLoader(TensorDataset(X_val), batch_size=256, shuffle=False)

        # 处理 X_train
        X_train_processed = []
        for batch in train_loader:
            X_train_processed.append(prepareDataDS(mdl1, batch[0]))  # 假设 prepareDataDS 只接受输入特征
        X_train = torch.cat(X_train_processed)

        # 处理 X_val
        X_val_processed = []
        for batch in val_loader:
            X_val_processed.append(prepareDataDS(mdl1, batch[0]))
        X_val = torch.cat(X_val_processed)

        print('prepareDataDS done')

        torch.cuda.empty_cache()
        del mdl1
        # 5️⃣ **转换 `X_train` 和 `X_val` 为 PyTorch Tensor**
        X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
        X_val = torch.tensor(X_val, dtype=torch.float32).permute(0, 2, 1)


        # 6️⃣ **确保 `Y_train['out']` 和 `Y_val['out']` 也是 Tensor**
        Y_train_out = Y_train['out'].clone().detach()
        Y_val_out = Y_val['out'].clone().detach()

        # 7️⃣ **创建 DataLoader**
        train_dataset = TensorDataset(X_train, Y_train_out)
        val_dataset = TensorDataset(X_val, Y_val_out)
        train_loader = DataLoader(train_dataset, batch_size=192, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=192, shuffle=False)


        model = model_dict[mdlName2](length).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        model_path = f'models/{mdlName2}_subject_normal_global_model2_fold{foldname}.pth'
        train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs=50, model_path=model_path, is_refinement=True)

        torch.cuda.empty_cache()  # 释放 GPU 内存

def train_mlp(mlp, unet, train_loader, val_loader, loss_fn, optimizer, epochs, model_path):
    """训练 MLP 以从 UNet 特征 level4 预测 ABP (SBP, DBP)，支持多 GPU 训练"""
    best_val_loss = float('inf')

    # 💡 **支持多 GPU**
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        mlp = nn.DataParallel(mlp)

    mlp.to(device)
    unet.eval()  # **冻结 UNet，不计算梯度**

    for epoch in range(epochs):
        mlp.train()
        train_loss = 0.0

        for batch in train_loader:
            X_batch, Y_batch = batch[0].to(device), batch[1].to(device)

            # **提取 SBP（最高值）和 DBP（最低值）**
            SBP = torch.max(Y_batch, dim=1)[0]  # 取每个样本的最大值
            DBP = torch.min(Y_batch, dim=1)[0]  # 取每个样本的最小值
            Y_target = torch.cat([SBP, DBP], dim=-1)  # 形状变为 (batch_size, 2)

            with torch.no_grad():  # **不计算梯度**
                outputs = unet(X_batch)
                mlp_features = outputs['level4'].view(outputs['level4'].size(0), -1)  # Flatten 维度

            optimizer.zero_grad()
            preds = mlp(mlp_features)  # MLP 预测 SBP, DBP
            loss = loss_fn(preds, Y_target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # **验证**
        mlp.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                X_batch, Y_batch = batch[0].to(device), batch[1].to(device)

                # **提取 SBP 和 DBP**
                SBP = torch.max(Y_batch, dim=1)[0]
                DBP = torch.min(Y_batch, dim=1)[0]
                Y_target = torch.cat([SBP, DBP], dim=-1)

                outputs = unet(X_batch)
                mlp_features = outputs['level4'].view(outputs['level4'].size(0), -1)

                preds = mlp(mlp_features)
                loss = loss_fn(preds, Y_target)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss / len(train_loader):.4f}, "
              f"Val Loss: {val_loss / len(val_loader):.4f}")

        # 💡 只在主 GPU 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if torch.cuda.device_count() > 1:
                torch.save(mlp.module.state_dict(), model_path)  # 多 GPU 训练
            else:
                torch.save(mlp.state_dict(), model_path)  # 单 GPU 训练


# 主函数
def main():
    # train_approximate_network()  # 训练近似模型
    # train_refinement_network()   # 训练细化模型
    # train_resnet_network()
    train_unet_network()

if __name__ == '__main__':
    main()
