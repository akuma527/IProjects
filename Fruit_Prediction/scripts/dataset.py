from torch.utils.data import Dataset
import cv2
import pandas as pd
import os
from PIL import Image
import config


class YourDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.samples = []
        self.classes = []
        self.transform = transform

        for fruit in os.listdir(data_root):
            fruit_folder = os.path.join(data_root, fruit)

            for name in os.listdir(fruit_folder+'/'):
                self.samples.append(os.path.join(fruit_folder, name))
                self.classes.append(fruit)
                
    def get_image_from_folder(self, name):
        """
        gets a image by a name gathered from file list text file

        :param name: name of targeted image
        :return: a PIL image
        """
        image = cv2.imread(name)
        myimage = Image.fromarray(image)
        return myimage

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name = self.samples[idx]
        X = self.get_image_from_folder(image_name)
        if self.transform is not None:
            X = self.transform(X)
        y = config.STR_2_INT[self.classes[idx]]
        return X, y

# if __name__ == '__main__':
#     dataset = YourDataset('data/Training/')
#     print(len(dataset))
#     print(dataset[4200])

