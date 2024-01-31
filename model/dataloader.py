import pandas as pd
import numpy as np
import nibabel as nib
import torch
import torch.utils.data as data

class load_data:
    def load_data():
        # parent_path = '/data/qneuromark/Results/ICA/UKBioBank'
        dic = {}
        data = []
        target = []
        file_name = '/data/users3/ygao11/data/Moder_Severe_AD/Data/UKBiobank/UKB_ICA_age_filepaths.csv'
        UKB_demo = pd.read_csv(file_name)
        
        ICN = pd.read_csv('/data/users3/ygao11/data/Moder_Severe_AD/Data/NeuroMark_info/53ICN_neuromark.csv')
        ICN = np.array(ICN["'GSP_IC_ID'"].to_list()) - 1
        
        for i,row in UKB_demo.iterrows():
            comp = nib.load(UKB_demo.iloc[i]['ICA_full_fname']).get_fdata()
            if comp.shape[0] > 490:
                comp = comp[:490]
            comp = comp[:,ICN]
            if comp.shape != (490, 53):
                raise ValueError(f"Expected shape (490, 53), but got {comp.shape}.")
            data.append(comp)
            target.append(UKB_demo.iloc[i]['age'])
            
        '''
        Undersample time series dataset 4x
        '''
        print(f"undersample time series dataset 4x")
        X = np.array(data)
        X = X[:,0:488,:]
        down_X = np.mean(X.reshape(X.shape[0],-1,4,53),axis=2)
        
        y = np.array(target)
            
        return down_X,y
    
class sliding_window_overlap:
    def sliding_windows_overlap(data):
        # print(f'the generative data is prepared in sliding window overlap approach')
        N, T, F = data.shape  # where T is the time dimension (490)
        window_size = 24
        overlap = 20
        step = window_size - overlap
        
        # Lists to store input and output sequences
        inputs = []
        outputs = []
        
        # Slide over the time dimension
        for t in range(0, T - window_size + 1, step):
            input_window = data[:, t:t+overlap, :]
            output_window = data[:, t+overlap:t+window_size, :]

            inputs.append(input_window.numpy())  # Convert to NumPy array
            outputs.append(output_window.numpy())
        
        # Convert lists to numpy arrays [25, n, 20/4,53] to [n,25,20/4,53] where 25 is the number of window size
        inputs = np.array(inputs).swapaxes(0, 1)
        outputs = np.array(outputs).swapaxes(0, 1)
        
        X_sli_train = inputs.reshape(-1,20,53)
        y_sli_train = outputs.reshape(-1,4,53)
        
        return X_sli_train, y_sli_train
    
            
class TimeSeriesDataset(data.Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = self.data[index]
        y = self.target[index]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
