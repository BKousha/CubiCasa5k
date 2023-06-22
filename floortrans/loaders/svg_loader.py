import lmdb
import pickle
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
#from numpy import genfromtxt
import pandas as pd
from floortrans.loaders.house import House


class FloorplanSVG(Dataset):
    def __init__(self, data_folder, data_file, is_transform=True,
                  img_norm=True):
        self.img_norm = img_norm
        self.is_transform = is_transform
        
        self.get_data = None
        self.get_data = self.get_txt
       
        self.data_folder = data_folder
        # Load txt file to list
        df = pd.read_csv(data_folder + data_file)
        print(df.columns)
        #print(df)
        self.files=df.iloc[:,0].values
        #print(self.files)

    def __len__(self):
        """__len__"""
        return len(self.files)

    def __getitem__(self, index):
        sample = self.get_data(index)

        if self.is_transform:
            sample = self.transform(sample)

        return sample

    def get_txt(self, index):
        filename=self.files[index]
        print(filename)
        fplan = cv2.imread(filename)
        fplan = cv2.cvtColor(fplan, cv2.COLOR_BGR2RGB)  # correct color channels
        height, width, nchannel = fplan.shape
        fplan = np.moveaxis(fplan, -1, 0)

        # # Getting labels for segmentation and heatmaps
        # house = House(self.data_folder + self.folders[index] + self.svg_file_name, height, width)
        # # Combining them to one numpy tensor
        # label = torch.tensor(house.get_segmentation_tensor().astype(np.float32))
        # heatmaps = house.get_heatmap_dict()
        coef_width = 1
        

        img = torch.tensor(fplan.astype(np.float32))

        sample = {'image': img, 'file': self.files[index],
                  'scale': coef_width}

        return sample

    

    def transform(self, sample):
        fplan = sample['image']
        # Normalization values to range -1 and 1
        fplan = 2 * (fplan / 255.0) - 1

        sample['image'] = fplan

        return sample
