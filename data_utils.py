import glob
import os

import cv2
import torch
import torch.utils.data as data

from helper import *


class TrainDataSegmentation(data.Dataset):
    def __init__(self, folder_path, transform=None):
        super(TrainDataSegmentation, self).__init__()
        self.folder_path = folder_path
        self.transform = transform
        self.img_files = glob.glob(os.path.join(self.folder_path, 'images', 'train', '*.jpg'))
        self.mask_files = []
        for img_path in self.img_files[:3500]:
            base = os.path.basename(img_path)
            filename = os.path.splitext(base)[0]
            if os.path.isfile(os.path.join(self.folder_path, 'labels', 'train', filename + '.png')):
                self.mask_files.append(os.path.join(self.folder_path, 'labels', 'train', filename + '.png'))
            else:
                self.img_files.remove(img_path)

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        image = plt.imread(img_path)[:, :, :3]  # type: np.array
        image = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
        image = np.moveaxis(image, -1, 0)
        image = torch.from_numpy(image).float()
        seg_image = plt.imread(mask_path)[:, :, :3]  # type: np.array
        seg_image = cv2.resize(seg_image, (640, 360), interpolation=cv2.INTER_AREA)
        label = image2label(seg_image)
        label = torch.from_numpy(label).long()
        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.mask_files)


class ValDataSegmentation(data.Dataset):
    def __init__(self, folder_path, transform=None):
        super(ValDataSegmentation, self).__init__()
        self.folder_path = folder_path
        self.img_files = glob.glob(os.path.join(self.folder_path, 'images', 'val', '*.jpg'))
        self.mask_files = []
        self.transform = transform
        for img_path in self.img_files:
            base = os.path.basename(img_path)
            filename = os.path.splitext(base)[0]
            if os.path.isfile(os.path.join(self.folder_path, 'labels', 'val', filename + '_train_color.png')):
                self.mask_files.append(os.path.join(self.folder_path, 'labels', 'val', filename + '_train_color.png'))
                # print("found label image")
            else:
                self.img_files.remove(img_path)
                # print("did not find label image")

    def __getitem__(self, index):
        if index < 0 or index >= 2000:
            print("index out of range {}".format(index))
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        image = plt.imread(img_path)[:, :, :3]  # type: np.array
        image = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
        image = np.moveaxis(image, -1, 0)
        image = torch.from_numpy(image).float()
        seg_image = plt.imread(mask_path)[:, :, :3]  # type: np.array
        seg_image = cv2.resize(seg_image, (640, 360), interpolation=cv2.INTER_AREA)
        label = image2label(seg_image)
        label = torch.from_numpy(label).long()

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.mask_files)


class TestDataSegmentation(data.Dataset):
    def __init__(self, folder_path, transform=None):
        super(TestDataSegmentation, self).__init__()
        self.folder_path = folder_path
        self.img_files = glob.glob(os.path.join(self.folder_path, 'images', 'test', '*.jpg'))
        self.mask_files = []
        self.transform = transform

    def __getitem__(self, index):
        if index < 0 or index >= 2000:
            print("index out of range {}".format(index))
        img_path = self.img_files[index]
        image = plt.imread(img_path)[:, :, :3]  # type: np.array
        image = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
        image = np.moveaxis(image, -1, 0)
        image = torch.from_numpy(image).float()

        if self.transform:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.mask_files)


class Normalize(object):
    def __init__(self):
        pass

    def __call__(self, image):
        image /= 255.
        return image
