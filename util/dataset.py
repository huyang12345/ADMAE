# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL
import random
from glob import glob
from torchvision import datasets, transforms
from PIL import Image
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch.utils.data
list = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 
        'hazelnut', 'leather','metal_nut', 'pill', 'screw', 
        'tile', 'toothbrush', 'transistor','wood', 'zipper']


def underep_data_sampler(train_root, test_root, sample_rate=1.):
    train_list = os.listdir(train_root)
    test_list = os.listdir(test_root)
    train_images = [os.path.join(train_root, img) for img in train_list]
    train_images = random.sample(train_images, int(len(train_images) * sample_rate))
    test_images = [os.path.join(test_root, img) for img in test_list]
    return train_images, test_images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    from torchvision import get_image_backend
    return pil_loader(path)


class MvtecDataLoader(torch.utils.data.Dataset):

    # constructor of the class
    def __init__(self, path, transform, normal_number=0, shuffle=False, mode=None, sample_rate=None):
        if sample_rate is None:
            raise ValueError("Sample rate = None")
        self.current_normal_number = normal_number
        self.transform = transform
        org_images = [os.path.join(path, img) for img in os.listdir(path)]
        if mode == "train":
            images = random.sample(org_images, int(len(org_images)*sample_rate))
        elif mode == "test":
            images = org_images
        else:
            raise ValueError("WDNMD")
        # print("ORG SIZE -> {}, SAMPLED SIZE -> {}".format(len(org_images), len(images)) )
        images = sorted(images)
        self.images = images

    def __getitem__(self, index):
        image_path = self.images[index]
        # label = image_path.split('/')[-1].split('.')[0]
        label = image_path.split('/')[-2]
        # data = Image.open(image_path)
        data = default_loader(image_path)

        # data = TF.adjust_contrast(data, contrast_factor=1.5)
        data = self.transform(data)
        return data, label

    def __len__(self):
        return len(self.images)


class MvtecDataLoader_avg(torch.utils.data.Dataset): # train all picture
    # constructor of the class
    def __init__(self, root_path, transform):
        self.transform = transform
        org_images = []
        for cls in os.listdir(root_path):
            if cls in list:
                org_image = [os.path.join(root_path, cls, 'train', 'good', img) for img in
                             os.listdir(os.path.join(root_path, cls, 'train', 'good'))]
                org_images.extend(org_image)
        images = org_images
        images = sorted(images)
        self.images = images

    def __getitem__(self, index):
        image_path = self.images[index]
        label = image_path.split('/')[-1]
        data = Image.open(image_path).convert("RGB")
        data = self.transform(data)
        return data, label

    def __len__(self):
        return len(self.images)

def build_dataset(is_train, args):#  for cifar10 cifar100

    transform = build_transform(is_train,args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)
    '''
    ImageFolder
    datasets form as follows
    root/dog/xxx.png
    root/dog/xxy.png
    root/dog/xxz.png   
    root/cat/123.png
    root/cat/nsdf3.png
    root/cat/asd932_.png  
    '''
    n = len(dataset)
    n_test = random.sample(range(0, n), n)
    test_set = torch.utils.data.Subset(dataset, n_test)

    return test_set



def build_transform(is_train,args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            interpolation='bicubic',
            mean=mean,
            std=std,
        )#数据增广
        print('the transform_train is {}'.format(transform))
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    # t.append(
    #     transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    # )#数据没有增广
    t.append(
        transforms.Resize(256),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.RandomHorizontalFlip())
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    print('the transform_test is {}'.format(t))
    return transforms.Compose(t)
def build_dataset_anomaly(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'test')
    dataset = datasets.ImageFolder(root, transform=transform)
    n = len(dataset)
    n_test = random.sample(range(0, n), n)#随机将n个数字打乱
    test_set = torch.utils.data.Subset(dataset, n_test)

    return test_set
