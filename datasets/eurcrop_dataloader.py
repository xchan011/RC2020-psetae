import torch
import torch.utils.data
import pandas as pd  
import os
import numpy as np
from numpy import genfromtxt
import tqdm
import h5py

# need to specify 1- Country 2- region
# eurocrops dataset: Country, region, parcel ID
# 31 classes

BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8','B9','B10', 'B11', 'B12',
       'B8A']
classes = [33111022, 33301010, 33101100, 33200000, 33301040, 33101011, 33101012, 33107000,
 33111023, 33106120, 33700000, 33101080, 33101060, 33109000, 33106080, 33112000,
 33106060, 33101041, 33101051, 33101021, 33106130, 33106050, 33103000, 33304000,
 33500000, 33101042, 33101022, 33106042, 33101032, 33101072, 33104000]       
NORMALIZING_FACTOR = 1e-4
PADDING_VALUE = -1

class EuroCropsDataset(torch.utils.data.Dataset):

    def __init__(self, root, partition, country ,region, validfraction=0.1):
        # assert (mode in ["trainvalid", "traintest"] and scheme=="random") or (mode is None and scheme=="blocks") # <- if scheme random mode is required, else None
        # assert scheme in ["random","blocks"]
        # assert partition in ["train","test","trainvalid","valid"]
        self.partition = partition

        self.validfraction = validfraction
        # self.scheme = scheme

        # ensure that different seeds are set per partition
        # seed += sum([ord(ch) for ch in partition])
        # np.random.seed(seed)
        # torch.random.manual_seed(seed)
        # self.mode = mode

        self.root = root
        if self.partition == "train":
            self.h5_file_path = os.path.join(self.root, "HDF5s", "train", country+"_train.h5")
        elif  self.partition == "valid":
            self.h5_file_path = os.path.join(self.root, "HDF5s", "train", country+"_train.h5")
        elif  self.partition == "test":    
            self.h5_file_path = os.path.join(self.root, "HDF5s", "test", country+"_test.h5")
        
        
        
        csv_file_name = 'demo_eurocrops_' + region + '.csv'
        if self.partition == "train":
            csv_file_path = os.path.join(self.root, "csv_labels", "train", csv_file_name)
        
        elif  self.partition == "valid":
            csv_file_path = os.path.join(self.root, "csv_labels", "train", csv_file_name)
        elif  self.partition == "test":    
            csv_file_path = os.path.join(self.root, "csv_labels", "test", csv_file_name)
        # self.read_ids = self.read_ids_random
        # self.classes = pd.read_csv(csv_file_path, index_col=0)
        
        

        

        self.labelsfile = pd.read_csv(csv_file_path, index_col=0)
        self.mapping = self.labelsfile.set_index("crpgrpc")
        self.classes = self.labelsfile["crpgrpc"].unique()
        self.crpgrpn = self.labelsfile.groupby("crpgrpc").first().crpgrpn.values
        self.nclasses = len(self.classes)

        self.region = region
        # self.partition = partition
        # self.data_folder = "{root}/csv/{region}".format(root=self.root, region=self.region)
        #self.batchsize = batchsize
        
        self.data = pd.read_hdf(self.h5_file_path, self.region)
        
        
        ids = list(self.data.index)

        validsize = int(len(ids) * self.validfraction)
    
        np.random.shuffle(ids)
        validids = ids[:validsize]
        trainids = ids[validsize:]


        X_temp = self.data.loc[np.sort(trainids)]
        # X_transpose = X_temp.T


        print("splitting {} ids in {} for training and {} for validation".format(len(ids), len(trainids), len(validids)))
        assert len(validids) + len(trainids) == len(ids)
        if self.partition == "train":
            self.ids = trainids
        elif self.partition == "valid":
            self.ids = validids
        elif self.partition == "test": 
            self.ids = ids
        
        # print(self)
        # if self.partition == "train":
        #     return trainids
        # if self.partition == "valid":
        #     return validids
        


        

    def __str__(self):
        return "Dataset {}. region {}. partition {}. X:{}, y:{}".format(self.root, self.region, self.partition,str(len(self.X)) +"x"+ str(self.X[0].shape), self.y.shape)

    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.partition == "train":
            spectral_data = self.data.loc[self.ids[idx]]
            self.id = self.ids[idx]
        elif self.partition == "valid":
            spectral_data = self.data.loc[self.ids[idx]]
            self.id = self.ids[idx]
        elif self.partition == "test":    
            spectral_data = self.data.loc[self.ids[idx]]
            self.id = self.ids[idx]
        
        crop_no = self.labelsfile.loc[self.id]['crpgrpc']
        label = self.labelsfile.loc[self.id]['crpgrpn']
        
        y_label = classes.index(int(crop_no))
        length = max(map(len,spectral_data))
        spectral_data_array = np.empty((0, length))

        for ii in range(spectral_data.shape[0]):
            test = np.array(spectral_data[ii])* NORMALIZING_FACTOR
            test.shape = (-1, len(test))
            spectral_data_array = np.concatenate((spectral_data_array,test))

        # print()

        X = torch.tensor(spectral_data_array).type(torch.FloatTensor)
        y= torch.from_numpy(np.array(crop_no)).type(torch.LongTensor)
        
        return {'data':X, 'label':y_label, 'ids':self.id, 'crop name':label}
            
def find_regions(root, country,  partition):
    # country options AT, SI, DK
    path = os.path.join(root, "HDF5s", partition, country+"_"+partition+".h5")
    h5_file = h5py.File(path, 'r')
    regions =  list(h5_file.keys())
    print('Available regions: {}'.format(regions))
    return regions
            
# if __name__=="__main__":
#     root = "/Users/ayshahchan/Desktop/ESPACE/thesis/codes/thesis/data"
#     region_train = find_regions(root, country='AT_T33UWP',  partition="train")
#     #classmapping = "/data/BavarianCrops/classmapping.isprs.csv"
#     train = EuroCropsDataset(root="/Users/ayshahchan/Desktop/ESPACE/thesis/codes/thesis/data",
#                          partition="train",
#                          country='AT_T33UWP',
#                          region=region_train[0]
#                          )
#     train_data = torch.utils.data.DataLoader(dataset=train,batch_size=20)
#     for i in range(len(train)):
        
    # test = EuroCropsDataset(root="/Users/ayshahchan/Desktop/ESPACE/thesis/codes/thesis/data",
    #                      region="holl",
    #                      partition="test",
    #                      scheme="blocks",
    #                      classmapping = classmapping,
    #                      samplet=50)

    # valid = EuroCropsDataset(root="/Users/ayshahchan/Desktop/ESPACE/thesis/codes/thesis/data",
    #                      country='AT',
    #                      region=region_train[0],
    #                      partition="valid",
    #                      samplet=50)

