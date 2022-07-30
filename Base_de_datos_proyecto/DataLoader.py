import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from database_generator import AlzheimerDataset
from torch.utils.data import DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_train_valid_loader(data_train_dir, data_valid_dir, batch_size,augment,random_seed,shuffle=True):
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010],)
    # define transforms
    if augment:
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),normalize,])
        valid_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),normalize,])
    else:
        train_transform = transforms.Compose([transforms.Resize((227,227)),transforms.ToTensor(),normalize,])
        valid_transform = transforms.Compose([transforms.Resize((227,227)),transforms.ToTensor(),normalize,])

    # load the dataset
    train_dataset = AlzheimerDataset(csv_file= "train_data.csv", root_dir= data_train_dir, transform = train_transform)
    valid_dataset = AlzheimerDataset(csv_file="valid_data.csv", root_dir= data_valid_dir, transform = valid_transform)

    num_train = len(train_dataset)
    indices = list(range(num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle = shuffle)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=shuffle)

    return (train_loader, valid_loader)

def get_test_loader(data_dir,batch_size,shuffle=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],)

    # define transform
    test_transform = transforms.Compose([transforms.Resize((227,227)),transforms.ToTensor(),normalize,])

    test_dataset = AlzheimerDataset(csv_file = "test_data.csv", root_dir= data_dir, transform = test_transform)
    test_loader = DataLoader(dataset= test_dataset, batch_size=batch_size, shuffle=shuffle)

    return test_loader


# CIFAR10 dataset 
train_loader, valid_loader = get_train_valid_loader(data_dir = './data', batch_size = 64, augment = False, random_seed = 1)

test_loader = get_test_loader(data_dir = './data', batch_size = 64)