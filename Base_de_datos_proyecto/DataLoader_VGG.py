import numpy as np
import torch
from torchvision import transforms
from database_generator import AlzheimerDataset
from torch.utils.data import DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_train_valid_loader(valid_csv, data_train_dir, data_valid_dir, batch_size,augment,random_seed,shuffle=True):
    normalize = transforms.Normalize((0.5), (0.5))
    # define transforms
    if augment:
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),normalize,])
        valid_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),normalize,])
    else:
        train_transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize((224,224)),transforms.ToTensor(),normalize,])
        valid_transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize((224,224)),transforms.ToTensor(),normalize,])

    # load the dataset
    valid_dataset = AlzheimerDataset(csv_file= valid_csv, root_dir= data_valid_dir, transform = valid_transform)
    train_dataset = AlzheimerDataset(csv_file= "train_data.csv", root_dir= data_train_dir, transform = train_transform)

    num_train = len(train_dataset)
    indices = list(range(num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle = shuffle)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=shuffle)

    return (train_loader, valid_loader)

def get_test_loader(data_test_dir,batch_size,shuffle=True):
    normalize = transforms.Normalize((0.5), (0.5))

    # define transform
    test_transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize((224,224)),transforms.ToTensor(),normalize,])

    test_dataset = AlzheimerDataset(csv_file = "test_data.csv", root_dir= data_test_dir, transform = test_transform)
    test_loader = DataLoader(dataset= test_dataset, batch_size=batch_size, shuffle=shuffle)

    return test_loader