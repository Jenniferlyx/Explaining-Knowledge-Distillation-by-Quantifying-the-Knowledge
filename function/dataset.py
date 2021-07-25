import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import random

class MyDataSet(Dataset):
    def __init__(self, root, model):
        self.root = root
        self.img_class = os.listdir(self.root)
        self.img_class.sort()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if model == 'alexnet':
            self.transformation = transforms.Compose([transforms.Resize((227, 227)), transforms.ToTensor(), normalize])
        else:
            self.transformation = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
        self.name_list = []
        self.label_list = []
        self.size = 0
        for i in range(len(self.img_class)):
            sub_root = self.root + '/' + self.img_class[i] + '/'
            images = os.listdir(sub_root)
            images.sort()
            for im in images:
                self.name_list.append(self.img_class[i] + '/' + im)
                self.label_list.append(i)
                self.size += 1

    def __getitem__(self, idx):
        label = self.label_list[idx]
        image = self.transformation(Image.open(self.root+'/'+self.name_list[idx]).convert('RGB'))

        return idx, image, label

    def __len__(self):
        return self.size


class SigmaDataSet(Dataset):
    def __init__(self, root, model):
        self.root = root
        self.img_class = os.listdir(self.root)
        self.img_class.sort()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if model == 'alexnet':
            self.transformation = transforms.Compose([transforms.Resize((227, 227)), transforms.ToTensor(), normalize])
        else:
            self.transformation = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
        self.name_list = []
        self.label_list = []
        self.size = 0
        for i in range(len(self.img_class)):
            sub_root = self.root + '/' + self.img_class[i] + '/'
            images = os.listdir(sub_root)
            images.sort()
            for im in images:
                self.name_list.append(self.img_class[i] + '/' + im)
                self.label_list.append(i)
                self.size += 1

    def __getitem__(self, idx):
        label = self.label_list[idx]
        image = self.transformation(Image.open(self.root+'/'+self.name_list[idx]).convert('RGB'))
        image_name = self.name_list[idx]
        return idx, image_name, image, label

    def __len__(self):
        return self.size


class AugmentDataSet(Dataset):
    def __init__(self, root, model):
        self.root = root
        self.img_class = os.listdir(self.root)
        self.img_class.sort()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if model == 'alexnet':
            self.transformation = transforms.Compose([transforms.Resize((256, 256)), transforms.RandomCrop((227, 227)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        else:
            self.transformation = transforms.Compose([transforms.Resize((256, 256)), transforms.RandomCrop((224, 224)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        self.name_list = []
        self.label_list = []
        self.size = 0
        for i in range(len(self.img_class)):
            sub_root = self.root + '/' + self.img_class[i] + '/'
            images = os.listdir(sub_root)
            images.sort()
            for im in images:
                self.name_list.append(self.img_class[i] + '/' + im)
                self.label_list.append(i)
                self.size += 1

    def __getitem__(self, idx):
        label = self.label_list[idx]
        image = self.transformation(Image.open(self.root+'/'+self.name_list[idx]).convert('RGB'))
        image_name = self.name_list[idx]
        return idx, image_name, image, label

    def __len__(self):
        return self.size


class SelectDataSet(Dataset):
    def __init__(self, root, model, sample_num=1000, seed=0):
        self.root = root
        self.img_class = os.listdir(self.root)
        self.img_class.sort()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if model == 'alexnet':
            self.transformation = transforms.Compose([transforms.Resize((227, 227)), transforms.ToTensor(), normalize])
        else:
            self.transformation = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
        self.name_list = []
        self.label_list = []
        self.size = 0
        for i in range(len(self.img_class)):
            sub_root = self.root +'/'+ self.img_class[i]+'/'
            images = os.listdir(sub_root)
            images.sort()
            for im in images:
                self.name_list.append(self.img_class[i]+'/' + im)
                self.label_list.append(i)
                self.size += 1

        id_list = list(range(self.size))
        random.seed(seed)
        random_id = random.sample(id_list, sample_num)
        self.name_list = [self.name_list[i] for i in random_id]
        self.label_list = [self.label_list[i] for i in random_id]
        self.size = len(random_id)

    def __getitem__(self, idx):
        label = self.label_list[idx]
        image = self.transformation(Image.open(self.root+'/'+self.name_list[idx]).convert('RGB'))
        image_name = self.name_list[idx]
        return idx, image_name, image, label

    def __len__(self):
        return self.size


class SupplementDataSet(Dataset):
    def __init__(self, root, model):
        self.root = root
        self.img_class = os.listdir(self.root)
        self.img_class.sort()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if model == 'alexnet':
            self.transformation = transforms.Compose([transforms.Resize((256, 256)), transforms.RandomCrop((227, 227)), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(30), transforms.ColorJitter(brightness=1, contrast=1, saturation=1, hue=0.5), transforms.ToTensor(), normalize])
        else:
            self.transformation = transforms.Compose([transforms.Resize((256, 256)), transforms.RandomCrop((224, 224)), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(30), transforms.ColorJitter(brightness=1, contrast=1, saturation=1, hue=0.5), transforms.ToTensor(), normalize])
        self.name_list = []
        self.label_list = []
        self.size = 0
        for i in range(len(self.img_class)):
            sub_root = self.root + '/' + self.img_class[i] + '/'
            images = os.listdir(sub_root)
            images.sort()
            for im in images:
                self.name_list.append(self.img_class[i] + '/' + im)
                self.label_list.append(i)
                self.size += 1

    def __getitem__(self, idx):
        label = self.label_list[idx]
        image = self.transformation(Image.open(self.root+'/'+self.name_list[idx]).convert('RGB'))
        image_name = self.name_list[idx]
        return idx, image_name, image, label

    def __len__(self):
        return self.size