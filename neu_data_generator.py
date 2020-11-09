# Max-Heinrich Laves
# Institute of Mechatronic Systems
# Leibniz Universität Hannover, Germany
# 2019

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
from skimage import color
from torchvision import transforms
import tqdm




class OCTDataset(Dataset):
    """
    Loads the Kermany OCT data set
    """

    def __init__(self, npz_dir, crop_to=(496, 496), resize_to=(224, 224), file_ext='.npz', color=False):
        """
        Given the root directory of the dataset, this function initializes the
        data set

        :param img_dir: List with paths of raw images
        """

        self._crop_to = crop_to
        self._resize_to = resize_to
        self._color = color
        self._npz_dir = npz_dir
        # self._class_list = glob(self._npz_dir + '/*')
        # self._class_list = sorted([c.split('/')[-1] for c in self._class_list])  # crop full paths/names.npz
        self._npz_file_names = []
        self._npz_file_names += sorted(glob(self._npz_dir + '/*'))

    def __len__(self):
        return len(self._npz_file_names)

    def __getitem__(self, idx):

        x = np.load(self._npz_file_names[idx])['data']
        y = np.load(self._npz_file_names[idx])['pos']
        if self._color:
            x = color.gray2rgb(x)

        x = np.atleast_3d(x)

        trans = transforms.Compose([  # zuheji
            transforms.ToPILImage(),
            #transforms.CenterCrop(self._crop_to),
            transforms.Resize(self._resize_to),
            transforms.ToTensor()
        ])

        x = trans(x)

        return x, y



if __name__ == "__main__":
    from matplotlib import pyplot as plt
    #dataset = OCTDataset('/home/mohanxu/Desktop/ophonlas-oct/oct_network_app/Data_neu100')
    dataset = OCTDataset('/home/mohanxu/Desktop/code_mohan2/Data_VS/Test')
    data_loader = DataLoader(dataset, batch_size=1, num_workers=1)
    batches = tqdm.tqdm(data_loader)
    for i_batch, (data,pos) in enumerate(batches):   #data.size() = [20,1,224,224]

        print(i_batch, data.size(), data.type())
        print(pos)
        #print(data)
        #print(img_pyramid[-1].size())
        plt.imshow(data.data.cpu().numpy()[0, 0])  # 256*256
        #print(target.data.cpu())
        plt.pause(2)
        #plt.clf()
