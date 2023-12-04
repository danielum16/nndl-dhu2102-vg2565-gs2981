# two_tier_classifier.py
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10  # Replace
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.models as models
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd


class AnimalDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None, is_super=True, use_gpu=False):
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform
        self.is_super = is_super
        self.use_gpu = use_gpu

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.df.iloc[idx, 0])
        # print('img name')
        # print(img_name)
        image = Image.open(img_name).convert("RGB")

        # Superclass label is in the second column, subclass label is the third column
        label_col_idx = 1
        if not self.is_super:
            label_col_idx = 2

        label = self.df.iloc[idx, label_col_idx]

        if self.transform:
            image = self.transform(image)

        if self.use_gpu:
            label = label.to('cuda')
            image = image.to('cuda')

        return image, label

class SuperclassModel(nn.Module):
    def __init__(self, num_superclasses):
        super(SuperclassModel, self).__init__()
        self.num_classes = num_superclasses
        self.base_model = models.densenet121(pretrained=False)
        # modify the last layer to match our data set superclass count
        self.base_model.classifier = nn.Linear(self.base_model.classifier.in_features, num_superclasses)

    def forward(self, x):
        return self.base_model(x)

class SubclassModel(nn.Module):
    def __init__(self, num_subclasses, num_superclasses):
        super(SubclassModel, self).__init__()
        self.num_classes = num_subclasses
        self.base_model = models.densenet121(pretrained=False)
        # modify the last layer to match our data set subclass count
        self.base_model.classifier = nn.Linear(self.base_model.classifier.in_features, num_subclasses)
        self.concatenation = nn.Linear(num_superclasses + num_subclasses, num_subclasses)
        # self.output_layer = nn.Linear(512, num_subclasses)

    def forward(self, x_image, x_superclass):
        if x_superclass.requires_grad is False:
            x_superclass.requires_grad = True
        base_out = self.base_model(x_image)
        concatenated_input = torch.cat((base_out, x_superclass), dim=1)
        # x = torch.relu(self.concatenation(concatenated_input))
        x = self.concatenation(concatenated_input)
        # return torch.softmax(self.output_layer(x), dim=1)
        return torch.softmax(x, dim=1)


def train_models(superclass_model, subclass_model, dataloader1, dataloader2, num_epochs_super, num_epochs_sub, use_gpu):
    print('Start training...')
    superclass_optimizer = optim.Adam(superclass_model.parameters(), lr=0.001)
    #if use_gpu:
    #    superclass_optimizer = superclass_optimizer.to('cuda')
    superclass_criterion = nn.CrossEntropyLoss()

    # Train the superclass model
    for epoch in range(num_epochs_super):
        total_loss = 0.0
        for inputs, labels in dataloader1:
            if use_gpu:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            superclass_optimizer.zero_grad()
            outputs = superclass_model(inputs)
            loss = superclass_criterion(outputs, labels)
            # print('training loss: ' + str(loss))
            loss.backward()
            superclass_optimizer.step()
            total_loss += loss.item()  # Accumulate the loss

        average_loss = total_loss / len(dataloader1)
        print(f'Epoch {epoch + 1}/{num_epochs_super}, Average Loss: {average_loss}')

    print('superclass training done')

    subclass_optimizer = optim.Adam(subclass_model.parameters(), lr=0.0001)
    #if use_gpu:
    #    subclass_optimizer = subclass_optimizer.to('cuda')
    subclass_criterion = nn.CrossEntropyLoss()

    # Train the subclass model
    for epoch in range(num_epochs_sub):
        total_loss = 0.0
        for inputs, labels in dataloader2:
            if use_gpu:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            superclass_outputs = superclass_model(inputs)
            subclass_optimizer.zero_grad()
            outputs = subclass_model(inputs, superclass_outputs)
            loss = subclass_criterion(outputs, labels)
            loss.backward()
            subclass_optimizer.step()
            total_loss += loss.item()  # Accumulate the loss
        average_loss = total_loss / len(dataloader2)
        print(f'Epoch {epoch + 1}/{num_epochs_sub}, Average Loss: {average_loss}')

    print('training complete')
    return superclass_model, subclass_model

if __name__ == "__main__":
    # Define parameters
    input_size = (3, 64, 64)
    num_superclasses = 3  # Number of superclasses (bird, dog, reptile)
    num_subclasses = 87  # Number of subclasses (87 classes)
    num_novel = 1  # Number of novel class
    epochs_super = 20
    epochs_sub = 50
    batch_size = 32
    # root_dir =
    train_data_dir = "./Released_Data/train_shuffle"
    csv_path = "./Released_Data/train_data.csv"
    use_gpu = False

    if torch.cuda.is_available():
        print("GPU is available!")
        use_gpu = True

    # Create dataset and dataloaders
    transform = transforms.Compose(
        [transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)  # Replace with actual dataset

    # df = pd.read_csv(csv_path)

    training_dataset_superclass = AnimalDataset(csv_path, train_data_dir, transform=transform, is_super=True)
    # train_dataset = ImageFolder(root=train_data_dir, transform=transform)
    dataloader1 = DataLoader(training_dataset_superclass, batch_size=batch_size, shuffle=True)

    training_dataset_subclass = AnimalDataset(csv_path, train_data_dir, transform=transform, is_super=False)
    # train_dataset = ImageFolder(root=train_data_dir, transform=transform)
    dataloader2 = DataLoader(training_dataset_subclass, batch_size=2*batch_size, shuffle=True)

    # Create and initialize the superclass model
    superclass_model = SuperclassModel(num_superclasses + num_novel)
    if use_gpu:
        superclass_model = superclass_model.to('cuda')

    # Create and initialize the subclass model
    subclass_model = SubclassModel(num_subclasses + num_novel, num_superclasses + num_novel)
    if use_gpu:
        subclass_model = subclass_model.to('cuda')

    # Train both models
    trained_superclass_model, trained_subclass_model = train_models(superclass_model, subclass_model, dataloader1,
                                                                    dataloader2, epochs_super, epochs_sub, use_gpu)

    # Save the trained models
    torch.save(trained_superclass_model.state_dict(), 'superclass_model.pth')
    torch.save(trained_subclass_model.state_dict(), 'subclass_model.pth')
