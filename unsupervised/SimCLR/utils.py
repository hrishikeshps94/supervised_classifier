from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10,ImageFolder
import cv2,tqdm
import numpy as np

class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target

class CustomDataset(ImageFolder):
    """ImageFolder Dataset.
    """

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)
        # img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target




train_transform = transforms.Compose([
    transforms.RandomResizedCrop(256),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),transforms.Normalize([0.392, 0.392, 0.392], [0.252, 0.252, 0.252])])

test_transform = transforms.Compose([transforms.Resize(512),
    transforms.ToTensor(),transforms.Normalize([0.392, 0.392, 0.392], [0.252, 0.252, 0.252])])
# transforms.Normalize([0.4134, 0.4134, 0.4134], [0.2701, 0.2701, 0.2701])]
# 0.39197805098829674 0.25170893252110943
def mean_and_std(file_list):
    print('Calculating mean and std of training set for data normalization.')
    m_list, s_list = [], []
    for img_filename in tqdm.tqdm(file_list):
        # img_filename = paths + str(i)
        img = cv2.imread(img_filename)
        (m, s) = cv2.meanStdDev(img)
        m_list.append(m.reshape((3,)))
        s_list.append(s.reshape((3,)))
    m_array = np.array(m_list)
    s_array = np.array(s_list)
    m = m_array.mean(axis=0, keepdims=True)
    s = s_array.mean(axis=0, keepdims=True)
    m = m[0][::-1][0]/255
    s = s[0][::-1][0]/255

    return m, s

# train_transform = transforms.Compose([
#     transforms.RandomResizedCrop(32),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
#     transforms.RandomGrayscale(p=0.2),
#     transforms.ToTensor(),
#     transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

# test_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])





