# """
# 	Miscellaneous helper functions
# """

# from tqdm import tqdm
# import numpy as np
# import matplotlib.pyplot as plt
# import pickle
# from metrics import *
# import seaborn as sns
# sns.set()
# length = 1024


# def prepareData(mdl, X_train, X_val, X_test, Y_train, Y_val, Y_test):
# 	"""
# 	Prepares data for 2nd stage training
	
# 	Arguments:
# 		mdl {keras.model} -- keras model
# 		X_train {array} -- X train
# 		X_val {array} -- X val
# 		X_test {array} -- X test
# 		Y_train {array} -- Y train
# 		Y_val {array} -- Y val
# 		Y_test {array} -- Y test
	
# 	Returns:
# 		tuple -- tuple of X_train, X_val and X_test for 2nd stage training
# 	"""
	
# 	X2_train = []

# 	X2_val = []

# 	X2_test = []


# 	YPs = mdl.predict(X_train)

# 	for i in tqdm(range(len(X_train))):

# 		X2_train.append(np.array(YPs[i]))



# 	YPs = mdl.predict(X_val)

# 	for i in tqdm(range(len(X_val))):

# 		X2_val.append(np.array(YPs[i]))



# 	YPs = mdl.predict(X_test)

# 	for i in tqdm(range(len(X_test))):

# 		X2_test.append(np.array(YPs[i]))


# 	X2_train = np.array(X2_train)

# 	X2_val = np.array(X2_val)

# 	X2_test = np.array(X2_test)

# 	return (X2_train, X2_val, X2_test)


# def prepareDataDS(mdl, X):
# 	"""
# 	Prepares data for 2nd stage training in the deep supervised pipeline
	
# 	Arguments:
# 		mdl {keras.model} -- keras model
# 		X {array} -- array being X train or X val
	
# 	Returns:
# 		X {array} -- suitable X for 2nd stage training
# 	"""
	
# 	X2 = []


# 	YPs = mdl.predict(X)
	
# 	for i in tqdm(range(len(X)),desc='Preparing Data for DS'):
		
# 	   X2.append(np.array(YPs[0][i]))

# 	X2 = np.array(X2)


# 	return X2


# def prepareLabel(Y):

# 	"""
# 	Prepare label for deep supervised pipeline
	
# 	Returns:
# 		dictionary -- dictionary containing the 5 level ground truth outputs of the network
# 	"""
	
# 	def approximate(inp,w_len):
# 		"""
# 		Downsamples using taking mean over window
		
# 		Arguments:
# 			inp {array} -- signal
# 			w_len {int} -- length of window
		
# 		Returns:
# 			array -- downsampled signal
# 		"""
		
# 		op = []
		
# 		for i in range(0,len(inp),w_len):
		
# 			op.append(np.mean(inp[i:i+w_len]))
			
# 		return np.array(op)
	
# 	out = {}
# 	out['out'] = []
# 	out['level1'] = []
# 	out['level2'] = []
# 	out['level3'] = []
# 	out['level4'] = []
	
	
# 	for y in tqdm(Y,desc='Preparing Label for DS'):
	
# 																	# computing approximations
# 		cA1 = approximate(np.array(y).reshape(length), 2)
		
# 		cA2 = approximate(np.array(y).reshape(length), 4)
		
# 		cA3 = approximate(np.array(y).reshape(length), 8)
		
# 		cA4 = approximate(np.array(y).reshape(length), 16)
		
		
		
# 																	# populating the labels for different labels
# 		out['out'].append(np.array(y.reshape(length,1)))
# 		out['level1'].append(np.array(cA1.reshape(length//2,1)))
# 		out['level2'].append(np.array(cA2.reshape(length//4,1)))
# 		out['level3'].append(np.array(cA3.reshape(length//8,1)))
# 		out['level4'].append(np.array(cA4.reshape(length//16,1)))
		
# 	out['out'] = np.array(out['out'])                                # converting to numpy array
# 	out['level1'] = np.array(out['level1'])
# 	out['level2'] = np.array(out['level2'])
# 	out['level3'] = np.array(out['level3'])
# 	out['level4'] = np.array(out['level4'])
	
	
# 	return out


import torch
from tqdm import tqdm
import numpy as np
length = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def prepareData(model, X_train, X_val, X_test):
    """
    Prepares data for 2nd stage training in PyTorch

    Arguments:
        model {torch.nn.Module} -- PyTorch model
        X_train {torch.Tensor} -- X train
        X_val {torch.Tensor} -- X val
        X_test {torch.Tensor} -- X test

    Returns:
        tuple -- tuple of X_train, X_val and X_test for 2nd stage training
    """

    model.eval()
    X_train, X_val, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_val, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
    X_train, X_val, X_test = X_train.to(device), X_val.to(device), X_test.to(device)

    with torch.no_grad():
        X2_train = model(X_train).detach().cpu().numpy()
        X2_val = model(X_val).detach().cpu().numpy()
        X2_test = model(X_test).detach().cpu().numpy()

    return X2_train, X2_val, X2_test

def prepareDataDS(model, X):
    """
    Prepares data for 2nd stage training in the deep supervised pipeline for PyTorch

    Arguments:
        model {torch.nn.Module} -- PyTorch model
        X {torch.Tensor} -- X train or X val

    Returns:
        X {torch.Tensor} -- suitable X for 2nd stage training
    """

    model.eval()
    X = X.to(device)

    with torch.no_grad():
        Y_preds = model(X)
        X2 = Y_preds['out'].detach().cpu()  # 取第一个输出（主输出层）

    return X2

def prepareLabel(Y):
    """
    Prepare label for deep supervised pipeline (PyTorch version)

    Returns:
        dictionary -- dictionary containing the 5 level ground truth outputs of the network
    """

    def approximate(inp, w_len):
        """Downsamples by taking mean over window"""
        return np.array([np.mean(inp[i:i+w_len]) for i in range(0, len(inp), w_len)])

    out = {
        'out': [],
        'level1': [],
        'level2': [],
        'level3': [],
        'level4': []
    }

    for y in Y:
        cA1 = approximate(np.array(y).reshape(length), 2)
        cA2 = approximate(np.array(y).reshape(length), 4)
        cA3 = approximate(np.array(y).reshape(length), 8)
        cA4 = approximate(np.array(y).reshape(length), 16)

        out['out'].append(np.array(y.reshape(length, 1)))
        out['level1'].append(np.array(cA1.reshape(length // 2, 1)))
        out['level2'].append(np.array(cA2.reshape(length // 4, 1)))
        out['level3'].append(np.array(cA3.reshape(length // 8, 1)))
        out['level4'].append(np.array(cA4.reshape(length // 16, 1)))

    out = {k: torch.tensor(np.array(v), dtype=torch.float32).to(device) for k, v in out.items()}
    return out

