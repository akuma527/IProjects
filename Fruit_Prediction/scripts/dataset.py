from torch.utils.data import Dataset
import cv2
import pandas as pd
import os


class YourDataset(Dataset):
    def __init__(self, data_root):
        self.samples = []

        for fruit in os.listdir(data_root):
            fruit_folder = os.path.join(data_root, fruit)

            with open(fruit_folder, 'r') as fruit_file:
                for name in fruit_file.read().splitlines():
                    self.samples.append((fruit, name))
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

if __name__ == '__main__':
    dataset = YourDataset('../data/Training/')
    print(len(dataset))
    print(dataset[420])

