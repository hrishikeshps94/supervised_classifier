import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset
import os,tqdm
import PIL.Image as Image
import numpy as np
import cv2
class Custom_Dataset(Dataset):


    def __init__(self,root_dir,transforms= None, is_train=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.is_train = is_train
        self.im_file_list = []
        self.class_list = []
        for dir_path, _, file_names in os.walk(root_dir):
            for f_paths in file_names:
                self.class_list.append(dir_path.split('/')[-1])
                self.im_file_list.append(os.path.join(dir_path,f_paths))
        self.class_names = sorted(os.listdir(root_dir))
        print(self.class_names)
        self.class_id_dict = {class_name:count for count,class_name in enumerate(self.class_names)}
        if is_train:
            self.train_transform = transforms
        elif not is_train:
            self.val_transform  = transforms
        # self.seg_norm = T.Normalize([0.193], [0.392])
        # self.normal_norm = T.Normalize([0.4134], [0.2701])
    def __len__(self):
        return len(self.im_file_list)
    def __getitem__(self, idx):
        image_fname = self.im_file_list[idx]
        image_class = self.class_list[idx]
        # edge_image_fname = image_fname.replace('normal','edge')
        rv_image_fname = image_fname.replace('normal','grading_seg')
        normal_im = Image.open(image_fname).convert('L')
        # edge_im = Image.open(edge_image_fname).convert('L')
        rv_im = Image.open(rv_image_fname).convert('L')
        comb_image = np.concatenate([np.asarray(normal_im)[None,None,...],np.asarray(rv_im)[None,None,...]],axis=0)
        comb_image = torch.from_numpy(comb_image)
        if self.is_train:
            comb_image = self.train_transform(comb_image)
        elif not self.is_train:
            comb_image = self.val_transform(comb_image)
        class_id = torch.tensor(self.class_id_dict[image_class])
        normal_im,rv_im = torch.tensor_split(comb_image,2,dim=0)
        normal_im,rv_im = normal_im/normal_im.max(),rv_im/rv_im.max()
        return normal_im.squeeze(0),rv_im.squeeze(0),class_id





