import torch
from torch import nn
from torchvision import models
import torch.optim as optim
from model import load_model
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from dataset import YourDataset
from torchvision.transforms import ToTensor, ToPILImage
from tqdm import tqdm
from torchvision import transforms
from torch.autograd import Variable


model = load_model()
criterion = nn.CrossEntropyLoss() 
optimizer = optim.SGD(model.classifier.parameters(), 
                      lr=0.001, 
                      momentum=0.9)


if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    print("Running on the GPU")


model.to(device)

# Define train transforms
train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],  # Mean for 3 channels (RGB) for colored images
                         [0.5, 0.5, 0.5])  # Std for 3 channels
])

# Define test transforms
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],  # 3 channels (RGB) for colored images
                         [0.5, 0.5, 0.5])
])


train_data = YourDataset(data_root='data/Training/', transform=train_transforms)
# test_data = YourDataset(data_root='data/Test/', testdata=True, transform=test_transforms)


# calculate size of train and validation sets
train_size = int(0.8 * len(train_data))
valid_size = len(train_data) - train_size
partial_train_ds, valid_ds = random_split(train_data, [train_size, valid_size])


# replace XX and YY with batch_size and number of workers, respectively
train_loader = DataLoader(partial_train_ds, batch_size=128, shuffle=True, num_workers=8)
valid_loader = DataLoader(valid_ds, batch_size=128, shuffle=True, num_workers=8)
# test_loader = DataLoader(test_data, batch_size=64, num_workers=8)


n_epochs = 20 # this is a hyperparameter you'll need to define
k = []
for epoch in (range(n_epochs)):
    ##################
    ### TRAIN LOOP ###
    ##################
    # set the model to train mode
    model.train()
    train_loss = 0
    total_train = 0
    total_val = 0
    correct_val = 0
    correct_train = 0
    for data in tqdm(train_loader):
        # clear the old gradients from optimized variables
        optimizer.zero_grad()
        # forward pass: feed inputs to the model to get outputs
        feature, target = data
        # target = torch.autograd.Variable(torch.tensor(target))
        feature = feature.view(-1, 3, 224, 224).to(device)
        # print(target)
        target = Variable(torch.tensor(target, dtype=torch.long)).to(device)
        output = model(feature)
        
        predicted = torch.argmax(output,1)
        # calculate the training batch loss
        loss = criterion(output, target)
        # backward: perform gradient descent of the loss w.r. to the model params
        loss.backward()
        # update the model parameters by performing a single optimization step
        optimizer.step()
        # accumulate the training loss
        total_train += target.size(0)
        # print(predicted, target)
        correct_train += (predicted == target).sum().item()
        
        train_loss += loss.item()
        # k+=1
        # print(k)

    #######################
    ### VALIDATION LOOP ###
    #######################
    # set the model to eval mode
    model.eval()
    valid_loss = 0
    # turn off gradients for validation
    with torch.no_grad():
        for data in tqdm(valid_loader):
            # forward pass
            feature, target = data
            feature = feature.view(-1, 3, 224, 224).to(device)
            target = Variable(torch.tensor(target, dtype=torch.long)).to(device)
            output = model(feature)
            
            # validation batch loss
            loss = criterion(output, target) 
            predicted = torch.argmax(output,1)
            total_val += target.size(0)
            correct_val += (predicted == target).sum().item()
            # accumulate the valid_loss
            valid_loss += loss.item()
            
    #########################
    ## PRINT EPOCH RESULTS ##
    #########################
    train_loss /= len(train_loader)
    valid_loss /= len(valid_loader)
    print(f'Epoch: {epoch+1}/{n_epochs}.. Training loss: {train_loss}.. Validation Loss: {valid_loss}')
    print(f'Training accuracy: {100 * correct_train / total_train}, Validation accuracy: {100 * correct_val / total_val}')

torch.save(model.state_dict(), 'py-model.model')