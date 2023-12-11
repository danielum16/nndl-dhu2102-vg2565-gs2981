# two_tier_classifier.py
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10  # Replace
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
#import torchvision
import torchvision.models as models
from PIL import Image
import time
import pandas as pd


class AnimalDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None, is_super=True, use_gpu=False):
        self.df = pd.read_csv(csv_path)
#        print('df length:')
#        print(len(self.df))        
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
        
#        if idx == 0:
#            print(f'image before transform in train: {image}')

        if self.transform:
            image = self.transform(image)
            
#        if idx < 7:
#            print(f'image {idx} after transform in train: {image}')
#            print(f'Dimension of the image {image.size()}')
##            print(f'idx: {idx}')
#            check_image_values(image, idx)

        if self.use_gpu:
            label = label.to('cuda')
            image = image.to('cuda')

        return image, label

def check_image_values(image, idx):
    total_count = 0
    positive_count = 0
    channels, height, width = image.shape
    for i in range(channels):
        for j in range(height):
            for k in range(width):
                total_count+=1
                value = image[i, j, k]
                if (value > 0):
                    positive_count+=1
    result = (float)(positive_count) / total_count
    print(f'The image {idx} value positive percentage {result}')

class AnimalTestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
#        self.super_map_df = super_map_df
#        self.sub_map_df = sub_map_df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self): # Count files in img_dir
        return len([fname for fname in os.listdir(self.img_dir)])

    def __getitem__(self, idx):
        img_name = str(idx) + '.jpg'
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

#        if idx == 0:
#            print(f'image before transform in test: {image}')

        if self.transform:
            image = self.transform(image)
            
#        if idx < 6:
#            print(f'image after transform in test: {image}')
#            print(f'Dimension of the image {image.size()}')
#            check_image_values(image)

        return image, img_name

class SuperclassModel(nn.Module):
    def __init__(self, num_superclasses, use_pretrained):
        super(SuperclassModel, self).__init__()
        self.num_classes = num_superclasses
        self.base_model = models.densenet121(pretrained=use_pretrained)
        # print(self.base_model)
        # modify the last layer to match our data set superclass count
        self.base_model.classifier = nn.Linear(self.base_model.classifier.in_features, num_superclasses)
        # print('Printing modified model')
        # print(self.base_model)

    def forward(self, x):
#        print(f'Dimension of the input x {x.size()}')
        return self.base_model(x)
        # return torch.softmax(self.base_model(x), dim=1)

class SubclassModel(nn.Module):
    def __init__(self, num_subclasses, num_superclasses, use_pretrained):
        super(SubclassModel, self).__init__()
        self.num_classes = num_subclasses
        self.base_model = models.densenet121(pretrained=use_pretrained)
        # modify the last layer to match our data set subclass count
        self.base_model.classifier = nn.Linear(self.base_model.classifier.in_features, num_subclasses)
        self.concatenation = nn.Linear(num_superclasses + num_subclasses, num_subclasses)
        # self.output_layer = nn.Linear(512, num_subclasses)

    def forward(self, x_image, x_superclass):
        if x_superclass.requires_grad is False:
            x_superclass.requires_grad = True
        base_out = self.base_model(x_image)
        concatenated_input = torch.cat((base_out, x_superclass), dim=1)
        # concatenated_input = torch.cat((torch.softmax(base_out, dim=1), x_superclass), dim=1)
        # x = torch.relu(self.concatenation(concatenated_input))
        x = self.concatenation(concatenated_input)
        # return torch.softmax(self.output_layer(x), dim=1)
        # return torch.softmax(x, dim=1)
        return x

def count_correct_predictions(outputs, labels):
    # print('count outputs: ')
    # print(outputs)
    softmax_outputs = nn.Softmax(dim=1)(outputs)
    # print('count softmax outputs: ')
    # print(softmax_outputs)
    count = 0
    _, predicted_labels = torch.max(softmax_outputs, 1)
    # print('predicted label size: ')
    # print(predicted_labels.size())
    # print('label size: ')
    # print(labels.size())
    for i in range(len(labels)):
        if labels[i] == predicted_labels[i]:
            count+=1
#        if labels[i] == 1:
#            print('it is category 1 which is a dog')
#        elif labels[i] == 2:
#            print('it is category 2 which is a reptile')
#        if predicted_labels[i] == 1:
#            print('Prediction is category 1 which is a dog')
#        elif predicted_labels[i] == 2:
#            print('Prediction is category 2 which is a reptile')
#    print('count_correct_predictions: count:')
#    print(count)
    return count

def train_models(superclass_model, subclass_model, dataloader1, dataloader2, num_epochs_super, num_epochs_sub, use_gpu):
    print('Start training...')
    superclass_optimizer = optim.Adam(superclass_model.parameters(), lr=0.001)
    #if use_gpu:
    #    superclass_optimizer = superclass_optimizer.to('cuda')
    superclass_criterion = nn.CrossEntropyLoss()
#    print(f'data loader 1 size: {len(dataloader1)}')

    # Train the superclass model
    for epoch in range(num_epochs_super):
        total_loss = 0.0
        total_correct = 0
        total_labels = 0
#        dataloader_idx = 0
        for inputs, labels in dataloader1:
#            print('dataloader 1 idx: ')
#            print(dataloader_idx)
#            dataloader_idx +=1
            # print('printing labels...')
            # print(labels)
            if use_gpu:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            superclass_optimizer.zero_grad()
            outputs = superclass_model(inputs)
#            print('checking outputs...')
#            print(outputs)
#            print('checking outputs dimension...')
#            print(outputs.size())
#            print('checking labels dimension...')
#            print(labels.size())
#            print(labels)
#            print('checking inputs dimension...')
#            print(inputs.size())
            loss = superclass_criterion(outputs, labels)
            # print('training loss: ' + str(loss))
            loss.backward()
            superclass_optimizer.step()
            total_loss += loss.item()  # Accumulate the loss
            total_correct += count_correct_predictions(outputs, labels) # Accumulate the correct predictions
            total_labels += len(labels)

        average_loss = total_loss / len(dataloader1)
        accuracy = total_correct / total_labels
        print(f'Epoch {epoch + 1}/{num_epochs_super}, Average Loss: {average_loss}, Accuracy: {accuracy}')

    print('superclass training done')

    subclass_optimizer = optim.Adam(subclass_model.parameters(), lr=0.0001)
    #if use_gpu:
    #    subclass_optimizer = subclass_optimizer.to('cuda')
    subclass_criterion = nn.CrossEntropyLoss()

    # Train the subclass model
    for epoch in range(num_epochs_sub):
        total_loss = 0.0
        total_correct = 0
        total_labels = 0
        for inputs, labels in dataloader2:
            if use_gpu:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            superclass_outputs = superclass_model(inputs)
            subclass_optimizer.zero_grad()
#            print('superclass outputs in training subclass...')
#            print(superclass_outputs)
            
#            superclass_outputs_softmax = torch.softmax(superclass_outputs, dim=1)
#            print('superclass softmax outputs in training subclass...')
#            print(superclass_outputs_softmax)
#            _, predicted = torch.max(superclass_outputs_softmax.data, 1)
#            print('superclass max outputs in training subclass...')
#            print(predicted)
            outputs = subclass_model(inputs, superclass_outputs)
            loss = subclass_criterion(outputs, labels)
            loss.backward()
            subclass_optimizer.step()
            total_loss += loss.item()  # Accumulate the loss
            total_correct += count_correct_predictions(outputs, labels) # Accumulate the correct predictions
            total_labels += len(labels)
            
        average_loss = total_loss / len(dataloader2)
        accuracy = total_correct / total_labels
        print(f'Epoch {epoch + 1}/{num_epochs_sub}, Average Loss: {average_loss}, Accuracy: {accuracy}')

    print('training complete')
    return superclass_model, subclass_model

def test(superclass_model, subclass_model, dataloader1, dataloader2, use_gpu, save_to_csv=False, return_predictions=False):
    # Evaluate on test set, in this simple demo no special care is taken for novel/unseen classes
    test_predictions = {'image': [], 'superclass_index': [], 'subclass_index': []}
    dataloaders = [dataloader1, dataloader2]
    models = [superclass_model, subclass_model]
    
    with torch.no_grad():
        superclass_out = []
        for i in range(len(dataloaders)):
            test_loader = dataloaders[i]
            model = models[i]
            print(f'Processing class {i}')
            for j, data in enumerate(test_loader):
                inputs = data[0]
                if use_gpu:
            	    inputs = inputs.to('cuda')
                img_name = data[1]
            
                if i == 0:
                    outputs = model(inputs)
                    superclass_out.append(outputs)
                else:
                    outputs = model(inputs, superclass_out[j])
                
#                print('Outputs ...')
#                print(outputs)
                outputs_softmax = torch.softmax(outputs, dim=1)
#                print('outputs softmax ...')
#                print(outputs_softmax)

                _, predicted = torch.max(outputs_softmax.data, 1)
#                print('Test printing predicted ...')
#                print(predicted)
#                if predicted[0] == 1:
#                    print('Test Prediction is category 1 which is a dog')
#                elif predicted[0] == 2:
#                    print('Test Prediction is category 2 which is a reptile')
            
                if i == 0:
                    test_predictions['image'].append(img_name[0])
                    test_predictions['superclass_index'].append(predicted.item())
                else:
                    test_predictions['subclass_index'].append(predicted.item())
            
    test_predictions = pd.DataFrame(data=test_predictions)
        
    if save_to_csv:
        test_predictions.to_csv('test_predictions_gs.csv', index=False)
        
    if return_predictions:
        return test_predictions

if __name__ == "__main__":
    # Define parameters
    num_superclasses = 3  # Number of superclasses (bird, dog, reptile)
    num_subclasses = 87  # Number of subclasses (87 classes)
    num_novel = 1  # Number of novel class
    epochs_super = 50
    epochs_sub = 250
    batch_size = 32
    save_model = False
    # root_dir =
    train_data_dir = "./Released_Data/train_shuffle"
    train_csv_path = "./Released_Data/train_data.csv"
    test_data_dir = "./Released_Data/test_shuffle"
    test_csv_path = "./Released_Data/test_data.csv"
    worker_count = 1
    use_gpu = False
    use_pretrained = True

    if torch.cuda.is_available():
        print("GPU is available!")
        use_gpu = True
        
#    print(torchvision.__version__)

    # Create dataset, augment dataset and create dataloaders
    transform_train = transforms.Compose(
        [transforms.Resize((64, 64)), transforms.RandomVerticalFlip(p=0.3), transforms.RandomHorizontalFlip(p=0.3), transforms.RandomRotation(15), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), transforms.RandomGrayscale(p=0.2), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
    # transforms.RandomErasing(p=0.3), transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        
    # train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)  # Replace with actual dataset

    # df = pd.read_csv(csv_path)

    training_dataset_superclass = AnimalDataset(train_csv_path, train_data_dir, transform=transform_train, is_super=True)
    # train_dataset = ImageFolder(root=train_data_dir, transform=transform)
    dataloader1_train = DataLoader(training_dataset_superclass, batch_size=batch_size, shuffle=True, num_workers=worker_count)

    training_dataset_subclass = AnimalDataset(train_csv_path, train_data_dir, transform=transform_train, is_super=False)
    # train_dataset = ImageFolder(root=train_data_dir, transform=transform)
    dataloader2_train = DataLoader(training_dataset_subclass, batch_size=2*batch_size, shuffle=True, num_workers=worker_count)

    # Create and initialize the superclass model
    superclass_model = SuperclassModel(num_superclasses + num_novel, use_pretrained)
    if use_gpu:
        superclass_model = superclass_model.to('cuda')

    # Create and initialize the subclass model
    subclass_model = SubclassModel(num_subclasses + num_novel, num_superclasses + num_novel, use_pretrained)
    if use_gpu:
        subclass_model = subclass_model.to('cuda')

    start_time = time.time()
    # Train both models
    trained_superclass_model, trained_subclass_model = train_models(superclass_model, subclass_model, dataloader1_train,
                                                                    dataloader2_train, epochs_super, epochs_sub, use_gpu)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")
    
    # Inference starts (test dataset)
    transform_test = transforms.Compose(
        [transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_dataset = AnimalTestDataset(test_data_dir, transform=transform_test)
    dataloader1_test = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=worker_count)

    dataloader2_test = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=worker_count) 

    test(trained_superclass_model, trained_subclass_model, dataloader1_test, dataloader2_test, use_gpu, save_to_csv=True)

    # Save the trained models
    if save_model:
        super_class_model_file = f'superclass_model-{use_pretrained}.pth'
        sub_class_model_file = f'subclass_model-{use_pretrained}.pth'
        torch.save(trained_superclass_model.state_dict(), super_class_model_file)
        torch.save(trained_subclass_model.state_dict(), sub_class_model_file)
    
    print("Inference done.")
