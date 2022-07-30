#%%
from pickletools import optimize
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from database_generator import AlzheimerDataset
#%%
#Hyperparameters

batch = 32
num_epochs = 10
num_classes = 3
in_channel = 3
learning_rate = 1e-3

#Load Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
train_dataset = AlzheimerDataset(csv_file= "train_data.csv", root_dir="train", transform = transforms.ToTensor())
train_loader = DataLoader(dataset=train_dataset, batch_size=batch, shuffle = True)

valid_dataset = AlzheimerDataset(csv_file="valid_data.csv", root_dir="validation", transform = transforms.ToTensor())
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch, shuffle=True)

test_dataset = AlzheimerDataset(csv_file = "test_data.csv", root_dir="test", transform = transforms.ToTensor())
test_loader = DataLoader(dataset= test_dataset, batch_size=batch, shuffle=True)
#%%
print(test_loader)
#%%
#Model 
"""
model = torchvision.models.googlenet(pretrained=True)
model.to(device)

#Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#Train network
for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device= device)
        targets = targets.to(device = device)

        scores = model(data)
        loss = criterion(scores, targets)

        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    print(f"Cost at {epoch} is {sum(losses)/len(losses)}")

def chech_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device = device)
            y = y.to(device = device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
print("Checking accuracy on trainig set")
chech_accuracy(train_loader, model)

print("Mirar en test")
chech_accuracy(test_loader, model)
"""