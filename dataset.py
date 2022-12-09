from torch.utils.data.dataset import Dataset
from torchvision import transforms
import numpy as np
import pandas as pd
import os
import PIL.Image as Image
import albumentations as A
import cv2

class MITfaces(Dataset):
    def __init__(self, csv_path):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        super(MITfaces, self).__init__()
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # Second column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[1:, 0])
        # Third column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[1:, 1])
        # # Third column is for an operation indicator
        # self.operation_arr = np.asarray(self.data_info.iloc[:, 2])
        # Calculate len
        self.data_len = len(self.data_info.index)
        self.imgs = self.image_arr
        self.labels = self.label_arr
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index-1]
        # Open image
        img_as_img = Image.open(single_image_name)
        img_as_img = img_as_img.resize((20, 20), Image.BILINEAR)

        # Check if there is an operation
        # operations = self.operations
        # # If there is a resize operation
        # if operations == 'resize':
        #     # Resize image by Bilinear interpolation to 20x20
        #     img_as_img = img_as_img.resize((20,20), Image.BILINEAR)
        #     pass

        # Get label(class) of the image based on the cropped pandas column
        # Convert label to binary
        single_image_label = int(self.label_arr[index - 1])
        # if single_image_label == 'non_face':
        #     single_image_label = 0
        # else:
        #     single_image_label = 1


        # Augment image
        # augimg_as_img = self.A_transform(image=np.array(img_as_img))['image']
        # Transform image to tensor and normalize it
        img_as_img = self.transform(img_as_img)

        # # Save image
        # if not os.path.exists('augmented'):
        #     os.makedirs('augmented')
        #
        # if index % 10 == 0:
        #     large_augimage = cv2.resize(augimg_as_img, (200, 200), interpolation=cv2.INTER_LINEAR)
        #     large_image = cv2.resize(np.array(img_as_img), (200, 200), interpolation=cv2.INTER_LINEAR)
        #     hstack = np.hstack([large_image,large_augimage])
        #     cv2.imwrite('augmented/{}.jpg'.format(index), hstack)

        return (img_as_img, single_image_label, single_image_name)

    def __len__(self):
        return self.data_len


class MITtrain(Dataset):
    def __init__(self, csv_path,train_indices):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        super(MITtrain, self).__init__()
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # First column contains the image paths
        self.imgs = np.asarray(self.data_info.iloc[1:, 0])[train_indices]
        # Second column is the labels
        self.labels = np.asarray(self.data_info.iloc[1:, 1])[train_indices]
        # Calculate len
        self.data_len = self.imgs.shape[0]
        # Transforms
        self.A_transform = A.Compose([
            A.RandomRotate90(),
            A.Transpose(),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.25, rotate_limit=45, p=.5),
            # A.Blur(blur_limit=1),
            # A.OpticalDistortion(),
            # A.GridDistortion(),
            # A.HueSaturationValue(),
            # A.RandomBrightnessContrast(),
        ])

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.imgs[index]
        # Open image
        img_as_img = Image.open(single_image_name)
        img_as_img = img_as_img.resize((20, 20), Image.BILINEAR)

        # Check if there is an operation
        # operations = self.operations
        # # If there is a resize operation
        # if operations == 'resize':
        #     # Resize image by Bilinear interpolation to 20x20
        #     img_as_img = img_as_img.resize((20,20), Image.BILINEAR)
        #     pass

        # Get label(class) of the image based on the cropped pandas column
        # Convert label to binary
        single_image_label = int(self.labels[index])
        # if single_image_label == 'non_face':
        #     single_image_label = 0
        # else:
        #     single_image_label = 1


        # Augment image
        augimg_as_img = self.A_transform(image=np.array(img_as_img))['image']
        # Transform image to tensor and normalize it
        augimg_as_tensor = self.transform(augimg_as_img)

        # Save image
        if not os.path.exists('augmented'):
            os.makedirs('augmented')

        if index % 10 == 0:
            large_augimage = cv2.resize(augimg_as_img, (200, 200), interpolation=cv2.INTER_LINEAR)
            large_image = cv2.resize(np.array(img_as_img), (200, 200), interpolation=cv2.INTER_LINEAR)
            hstack = np.hstack([large_image,large_augimage])
            cv2.imwrite('augmented/{}.jpg'.format(index), hstack)

        return (augimg_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len

class MITval(Dataset):
    def __init__(self, csv_path,indices):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        super(MITval, self).__init__()
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # Second column contains the image paths
        self.imgs = np.asarray(self.data_info.iloc[1:, 0])[indices]
        # Third column is the labels
        self.labels = np.asarray(self.data_info.iloc[1:, 1])[indices]
        # # Third column is for an operation indicator
        # self.operation_arr = np.asarray(self.data_info.iloc[:, 2])
        # Calculate len
        self.data_len = self.imgs.shape[0]
        # Transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.imgs[index]
        # Open image
        img_as_img = Image.open(single_image_name)
        img_as_img = img_as_img.resize((20, 20), Image.BILINEAR)
        img_as_tensor = self.transform(img_as_img)

        single_image_label = int(self.labels[index])

        return (img_as_tensor, single_image_label )

    def __len__(self):
        return self.data_len



if __name__ == "__main__":
    # Call dataset
     dataset =  \
        MITfaces('./data/train/train.csv')


