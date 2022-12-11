from torch.utils.data.dataset import Dataset
from torchvision import transforms
import numpy as np
import pandas as pd
import os
import PIL.Image as Image
import albumentations as A
import cv2
from imgaug import augmenters as iaa
import imgaug as ia
from timm.data.auto_augment import rand_augment_transform

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
        self.imgs = np.asarray(self.data_info.iloc[1:, 0])
        # Third column is the labels
        self.labels = np.asarray(self.data_info.iloc[1:, 1])
        # Calculate length of data set
        self.data_len = len(self.data_info.index)
        # self.imgs = self.image_arr
        # self.labels = self.label_arr
        # Normalize images and convert to tensors
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.imgs[index-1]

        # Open image
        img_as_img = Image.open(single_image_name)
        img_as_img = img_as_img.resize((20, 20), Image.BILINEAR)
        single_image_label = int(self.labels[index-1])

        # Transform image into a normalized tensor
        img_as_img = self.transform(img_as_img)

        return (img_as_img, single_image_label, single_image_name)

    def __len__(self):
        return self.data_len

# Train dataset with balanced batch sampler
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
            A.ShiftScaleRotate(shift_limit=0.12, scale_limit=(-0.5, 0.5), rotate_limit=45, p=.75),
            A.Blur(blur_limit=3,p=1.0),
            A.OpticalDistortion(),
            A.GridDistortion(),
            A.HueSaturationValue(),
            A.RandomBrightnessContrast(),
        ])
        self.A_transform2 = A.Compose([
            A.RandomRotate90(),
            A.Flip(),
            A.Transpose(),
            A.OneOf([
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
            ], p=1.0),
            A.OneOf([
                A.MotionBlur(p=.75),
                A.MedianBlur(blur_limit=5, p=0.75),
                A.Blur(blur_limit=5, p=0.75),
            ], p=1.0),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.5, rotate_limit=45, p=0.75),
            A.OneOf([
                A.OpticalDistortion(p=0.75),
                A.GridDistortion(p=.75),
                A.IAAPiecewiseAffine(p=0.75),
            ], p=1.0),
            A.OneOf([
                A.CLAHE(clip_limit=4),
                A.IAASharpen(),
                A.IAAEmboss(),
                A.RandomBrightnessContrast(),
            ], p=1.0),
            A.HueSaturationValue(p=0.75),
        ])

        self.tfm = rand_augment_transform(config_str='rand-m9-n4-mstd0.5-inc1',
                                          hparams={'img_mean': [0.485, 0.456, 0.406],
                                                   'img_std': [0.229, 0.224, 0.225]} )

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    # def augmentor(self, image):
    #     'Apply data augmentation'
    #     sometimes = lambda aug: iaa.Sometimes(1.0, aug)
    #     seq = iaa.Sequential(
    #         [
    #             # apply the following augmenters to most images
    #             iaa.Fliplr(0.3),  # horizontally flip 10% of all images
    #             sometimes(iaa.Affine(
    #                 scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
    #                 # scale images to 80-120% of their size, individually per axis
    #                 translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
    #                 # translate by -20 to +20 percent (per axis)
    #                 rotate=(-45, 45),  # rotate by -45 to +45 degrees
    #                 shear=(-16, 16),  # shear by -16 to +16 degrees
    #                 order=[0, 1],
    #                 # use nearest neighbour or bilinear interpolation (fast)
    #                 cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
    #                 mode=ia.ALL
    #                 # use any of scikit-image's warping modes (see 2nd image from the top for examples)
    #             )),
    #             # execute 0 to 3 of the following (less important) augmenters per image
    #             # don't execute all of them, as that would often be way too strong
    #             iaa.SomeOf((3, 5), [
    #                 # [sometimes(iaa.Superpixels(p_replace=(0, 1.0),
    #                 #                            n_segments=(20, 200))),
    #                 # convert images into their superpixel representation
    #                 iaa.OneOf([
    #                     iaa.GaussianBlur((0, 3.0)),
    #                     # blur images with a sigma between 0 and 3.0
    #                     iaa.AverageBlur(k=(1, 7)),
    #                     # blur image using local means with kernel sizes between 2 and 7
    #                     iaa.MedianBlur(k=(1, 7)),
    #                     # blur image using local medians with kernel sizes between 2 and 7
    #                 ]),
    #                 iaa.Sharpen(alpha=(0, 1.0), lightness=(0.5, 1.5)),
    #                 # sharpen images
    #                 iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
    #                 # emboss images
    #                 # search either for all edges or for directed edges,
    #                 # blend the result with the original image using a blobby mask
    #                 iaa.BlendAlphaSimplexNoise(iaa.OneOf([
    #                     iaa.EdgeDetect(alpha=(0.5, 1.0)),
    #                     iaa.DirectedEdgeDetect(alpha=(0.5, 1.0),
    #                                            direction=(0.0, 1.0)),
    #                 ])),
    #                 iaa.AdditiveGaussianNoise(loc=0,
    #                                           scale=(0.0, 0.01 * 255),
    #                                           per_channel=0.5),
    #                 # add gaussian noise to images
    #                 iaa.OneOf([
    #                     iaa.Dropout((0.01, 0.1), per_channel=0.5),
    #                     # randomly remove up to 10% of the pixels
    #                     iaa.CoarseDropout((0.01, 0.1),
    #                                       size_percent=(0.01, 0.02),
    #                                       per_channel=0.2),
    #                 ]),
    #                 iaa.Invert(0.03, per_channel=True),
    #                 # invert color channels
    #                 iaa.Add((-10, 10), per_channel=0.5),
    #                 # change brightness of images (by -10 to 10 of original value)
    #                 iaa.AddToHueAndSaturation((-1, 1)),
    #                 # change hue and saturation
    #                 # either change the brightness of the whole image (sometimes
    #                 # per channel) or change the brightness of subareas
    #                 iaa.OneOf([
    #                     iaa.Multiply((0.8, 1.2), per_channel=0.5),
    #                     iaa.BlendAlphaFrequencyNoise(
    #                         exponent=(-1, 0),
    #                         foreground=iaa.Multiply((0.8, 1.2),
    #                                                 per_channel=True),
    #                         background=iaa.LinearContrast(
    #                             (0.9, 1.1))
    #                     )
    #                 ]),
    #                 sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5),
    #                                                     sigma=0.25)),
    #                 # move pixels locally around (with random strengths)
    #                 sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.1))),
    #                 # sometimes move parts of the image around
    #                 sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
    #             ], random_order=True)
    #         ], random_order=True)
    #
    #     return seq.augment_image(image)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.imgs[index]
        # Open image
        img_as_img = Image.open(single_image_name)
        img_as_img = img_as_img.resize((20, 20), Image.BILINEAR)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = int(self.labels[index])

        # Augment image
        augimg_as_img = self.A_transform(image=np.array(img_as_img))['image']
        # augimg_as_img = self.augmentor(image=np.array(img_as_img).astype('uint8'))
        # augimg_as_img = self.tfm(img_as_img)

        # Transform image into a normalized tensor
        augimg_as_tensor = self.transform(augimg_as_img)


        # Save horizontal stacked original image with their augmentations for tracking purposes during training
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

# Validation dataset
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
        # Calculate length of data set
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


