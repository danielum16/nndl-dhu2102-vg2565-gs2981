# two_tier_classifier.py
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.models as models
from PIL import Image
import time
import argparse
import pandas as pd
import matplotlib.pyplot as plt


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


def check_image_values(image, idx):
    total_count = 0
    positive_count = 0
    channels, height, width = image.shape
    for i in range(channels):
        for j in range(height):
            for k in range(width):
                total_count += 1
                value = image[i, j, k]
                if (value > 0):
                    positive_count += 1
    result = (float)(positive_count) / total_count
    print(f'The image {idx} value positive percentage {result}')


class AnimalTestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):  # Count files in img_dir
        return len([fname for fname in os.listdir(self.img_dir)])

    def __getitem__(self, idx):
        img_name = str(idx) + '.jpg'
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, img_name


class SuperclassModel(nn.Module):
    def __init__(self, num_superclasses, use_pretrained, model_idx):
        super(SuperclassModel, self).__init__()
        self.num_classes = num_superclasses
        model_map = {0: models.densenet121(pretrained=use_pretrained), 1: models.mobilenet_v3_small(pretrained=True),
                     2: models.squeezenet1_1(pretrained=True)}
        self.base_model = model_map[model_idx]
        # print(self.base_model)
        print(f'The current superclass backbone model type is: {self.base_model.__class__.__name__}')
        # modify the last layer to match our data set superclass count
        if model_idx == 0:
            self.base_model.classifier = nn.Linear(self.base_model.classifier.in_features, num_superclasses)
        elif model_idx == 1:
            self.base_model.classifier[-1] = nn.Linear(self.base_model.classifier[-1].in_features, num_superclasses)
        elif model_idx == 2:
            # Extract the in_channels of the last convolutional layer in the features
            final_conv = list(self.base_model.children())[-1]
            in_channels = final_conv[1].in_channels  # Get the current number of output channels
            # Replace the classifier
            self.base_model.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Conv2d(in_channels, num_superclasses, kernel_size=1),
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            )

    def forward(self, x):
        temp = self.base_model(x)
        return temp


class SubclassModel(nn.Module):
    def __init__(self, num_subclasses, num_superclasses, use_pretrained, model_idx):
        super(SubclassModel, self).__init__()
        self.num_classes = num_subclasses
        model_map = {0: models.densenet121(pretrained=use_pretrained), 1: models.mobilenet_v3_small(pretrained=True),
                     2: models.squeezenet1_1(pretrained=True)}
        self.base_model = model_map[model_idx]
        print(f'The current subclass backbone model type is: {self.base_model.__class__.__name__}')
        
        if model_idx == 0:
            self.base_model.classifier = nn.Linear(self.base_model.classifier.in_features, num_subclasses)
        elif model_idx == 1:
            self.base_model.classifier[-1] = nn.Linear(self.base_model.classifier[-1].in_features, num_subclasses)
        elif model_idx == 2:
            final_conv = list(self.base_model.children())[-1]
            in_channels = final_conv[1].in_channels  # Get the current number of output channels
            self.base_model.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Conv2d(in_channels, num_subclasses, kernel_size=1),
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            )

        self.concatenation = nn.Linear(num_superclasses + num_subclasses, num_subclasses)

    def forward(self, x_image, x_superclass):
        if x_superclass.requires_grad is False:
            x_superclass.requires_grad = True
        base_out = self.base_model(x_image)
        concatenated_input = torch.cat((base_out, x_superclass), dim=1)
        x = self.concatenation(concatenated_input)
        return x


def get_args():
    parser = argparse.ArgumentParser(description="Training parameters for your model")
    # Define arguments for each parameter
    parser.add_argument("--num_superclasses", type=int, default=3, help="Number of superclasses")
    parser.add_argument("--num_subclasses", type=int, default=87, help="Number of subclasses")
    parser.add_argument("--num_novel", type=int, default=0, help="Number of novel classes")
    parser.add_argument("--epochs_super", type=int, default=1, help="Number of epochs for superclass training")
    parser.add_argument("--epochs_sub", type=int, default=2, help="Number of epochs for subclass training")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--backbone_model", type=int, default=0,
                        help="Pick one backbone model, 0: DenseNet, 1: MobileNet, 2: SqueezeNet")
    parser.add_argument("--lr_super", type=float, default=0.001, help="Learning rate for superclass training")
    parser.add_argument("--lr_sub", type=float, default=0.0001, help="Learning rate for subclass training")
    parser.add_argument("--early_stop", action="store_true", default=False, help="Enable early stopping")
    parser.add_argument("--validation_pct", type=float, default=0.2, help="Validation dataset percentage")
    parser.add_argument("--wdecay_super", type=float, default=0.0001, help="Weight decay for superclass training")
    parser.add_argument("--wdecay_sub", type=float, default=0.001, help="Weight decay for subclass training")
    parser.add_argument("--save_model", action="store_true", default=False, help="Save the trained model")
    parser.add_argument("--plot", action="store_true", default=False,
                        help="Plot the loss and accuracy vs epochs graphs")
    parser.add_argument(
        "--superclass_model_path", type=str, help="Path to the pre-trained superclass model")
    parser.add_argument(
        "--subclass_model_path", type=str, help="Path to the pre-trained subclass model")

    # Parse the arguments and access them
    args = parser.parse_args()
    return args


def split_dataset(dataset, validation_pct):
    dataset_size = len(dataset)
    # print(f'dataset size: {dataset_size}')
    validation_size = int(validation_pct * dataset_size)
    training_size = dataset_size - validation_size

    # Use random_split to split the dataset
    training_dataset, validation_dataset = random_split(dataset, [training_size, validation_size])
    print(f'len of trainig {len(training_dataset)}, len of validation: {len(validation_dataset)}')
    return training_dataset, validation_dataset


def count_correct_predictions(outputs, labels):
    softmax_outputs = nn.Softmax(dim=1)(outputs)
    count = 0
    _, predicted_labels = torch.max(softmax_outputs, 1)
    for i in range(len(labels)):
        if labels[i] == predicted_labels[i]:
            count += 1
    return count


def train_model(model, optimizer, scheduler, criterion, dataloader, dataloader_val=None, superclass_model=None,
                num_epochs=20,
                early_stop=True, use_gpu=True, plot_graph=False):
    best_val_loss = float('inf')
    current_patience = 0
    patience = 10

    training_loss_data = []  # List of loss data for each hyperparameter
    validation_loss_data = []
    training_accuracy_data = []  # List of accuracy data for each hyperparameter

    is_super = True
    if superclass_model is not None:
        is_super = False

    if superclass_model is not None:
        superclass_model.eval()

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_correct = 0
        total_labels = 0

        model.train()
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            if use_gpu:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

            if is_super:
                # The model itself is superclass
                outputs = model(inputs)
            else:
                # The model is subclass
                super_out = superclass_model(inputs)
                outputs = model(inputs, super_out)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += count_correct_predictions(outputs, labels)
            total_labels += len(labels)

        average_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_labels
        scheduler.step(average_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # For plot
        training_loss_data.append(average_loss)
        training_accuracy_data.append(accuracy)

        avg_val_loss = 0.0
        if dataloader_val is not None:
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for inputs_val, labels_val in dataloader_val:
                    if use_gpu:
                        inputs_val, labels_val = inputs_val.to('cuda'), labels_val.to('cuda')

                    if is_super:
                        # The model itself is superclass
                        outputs_val = model(inputs_val)
                    else:
                        # The model is subclass
                        super_out_val = superclass_model(inputs_val)
                        outputs_val = model(inputs_val, super_out_val)

                    loss_val = criterion(outputs_val, labels_val)
                    total_val_loss += loss_val.item()

            model.train()
            avg_val_loss = total_val_loss / len(dataloader_val)

            validation_loss_data.append(avg_val_loss)

            if epoch > 20:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    current_patience = 0
                else:
                    current_patience += 1

            if current_patience >= patience and early_stop:
                print(f'Early stopping at epoch {epoch + 1}')
                break

        print(
            f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss}, Validation Loss: {avg_val_loss}. Accuracy: {accuracy}, Learning rate: {current_lr}')

    return model, training_loss_data, training_accuracy_data, validation_loss_data


def train_models(superclass_model, subclass_model, dataloader1, dataloader2, num_epochs_super, num_epochs_sub, lr_super,
                 lr_sub, wd_super, wd_sub, use_gpu, dataloader1_val, dataloader2_val, plot):
    print('Start training...')
    training_loss_data_t, training_accuracy_data_t, validation_loss_data_t = {super_key: [], sub_key: []}, {
        super_key: [], sub_key: []}, {super_key: [], sub_key: []}

    superclass_optimizer = optim.Adam(superclass_model.parameters(), lr=lr_super, weight_decay=wd_super)
    superclass_scheduler = ReduceLROnPlateau(superclass_optimizer, patience=5)
    superclass_criterion = nn.CrossEntropyLoss()

    print(f'data loader 1 size: {len(dataloader1)}')

    trained_superclass_model, training_loss_data_super, training_accuracy_data_super, validation_loss_data_super = train_model(
        superclass_model, superclass_optimizer, superclass_scheduler,
        superclass_criterion, dataloader1, dataloader_val=dataloader1_val,
        num_epochs=num_epochs_super, early_stop=early_stop, use_gpu=use_gpu, plot_graph=plot)
    print('Superclass training done. Subclass training starts...')

    subclass_optimizer = optim.Adam(subclass_model.parameters(), lr=lr_sub, weight_decay=wd_sub)
    subclass_scheduler = ReduceLROnPlateau(subclass_optimizer, patience=5)
    subclass_criterion = nn.CrossEntropyLoss()

    trained_subclass_model, training_loss_data_sub, training_accuracy_data_sub, validation_loss_data_sub = train_model(
        subclass_model, subclass_optimizer, subclass_scheduler, subclass_criterion,
        dataloader2, dataloader_val=dataloader2_val,
        superclass_model=trained_superclass_model, num_epochs=num_epochs_sub,
        early_stop=early_stop, use_gpu=use_gpu, plot_graph=plot)

    print('training complete')
    trained_models = [trained_superclass_model, trained_subclass_model]
    training_loss_data_t[super_key] = training_loss_data_super
    training_loss_data_t[sub_key] = training_loss_data_sub
    training_accuracy_data_t[super_key] = training_accuracy_data_super
    training_accuracy_data_t[sub_key] = training_accuracy_data_sub
    validation_loss_data_t[super_key] = validation_loss_data_super
    validation_loss_data_t[sub_key] = validation_loss_data_sub

    return trained_models, training_loss_data_t, training_accuracy_data_t, validation_loss_data_t


def predict_class(outputs, threshold=0.5, use_dynamic_threshold=False, mean_factor=1.0, std_factor=1.0):
    # Apply softmax to the outputs
    novel_idx = len(outputs[0])
    outputs_softmax = torch.softmax(outputs, dim=1)

    if use_dynamic_threshold:
#        print('using dynamic threshold...')
        # Calculate the mean and standard deviation of probabilities
        mean_prob = torch.mean(outputs_softmax)
        std_prob = torch.std(outputs_softmax)
        # Set the threshold based on mean and standard deviation
        threshold = mean_factor * mean_prob.item() + std_factor * std_prob.item()

    # print(threshold)

    # Check if none of the predicted probabilities exceed the threshold
    max_probs, predicted_class = torch.max(outputs_softmax, dim=1)
#    print('max probs ...')
#    print(len(max_probs))
#    print(max_probs)

    for i in range(len(predicted_class)):
        max_p = max_probs[i]
        if max_p.item() < threshold:
            predicted_class[i] = novel_idx
    return predicted_class


def add_to_prediction(c0, c1, img_names, predicted):
    test_predictions = {c0: [], c1: []}
    if len(img_names) != len(predicted):
        print('error, image size not equal to the number of predictions')
        return test_predictions

    for i in range(len(img_names)):
        img_name = img_names[i]
        predicted_class = predicted[i]
        test_predictions[c0].append(img_name)
        test_predictions[c1].append(predicted_class.item())
    return test_predictions


def test(superclass_model, subclass_model, dataloader, use_gpu, model_idx, save_to_csv=False, return_predictions=False):
    # Evaluate on test set, in this simple demo no special care is taken for novel/unseen classes
    column0 = 'ID'
    column1 = 'Target'
    superclass_model.eval()
    subclass_model.eval()
    test_predictions_super = {column0: [], column1: []}
    test_predictions_sub = {column0: [], column1: []}
    test_predictions = [test_predictions_super, test_predictions_sub]
    models = [superclass_model, subclass_model]

    with torch.no_grad():
        superclass_out = []
        for i in range(len(models)):
            model = models[i]
            current_batch = 0
            print(f'Processing class {i}')
            for inputs, img_names in dataloader:
                if use_gpu:
                    inputs = inputs.to('cuda')

                if i == 0:
                    outputs = model(inputs)
                    superclass_out.append(outputs)
                    predicted = predict_class(outputs, use_dynamic_threshold=True, std_factor=1.4)
                else:
                    outputs = model(inputs, superclass_out[current_batch])
                    predicted = predict_class(outputs, threshold=0.95, use_dynamic_threshold=False, std_factor=3)

                temp_predictions = add_to_prediction(column0, column1, img_names, predicted)
                test_predictions[i][column0].extend(temp_predictions[column0])
                test_predictions[i][column1].extend(temp_predictions[column1])
                current_batch += 1

    test_predictions_super = pd.DataFrame(data=test_predictions[0])
    test_predictions_sub = pd.DataFrame(data=test_predictions[1])

    if save_to_csv:
        test_predictions_super.to_csv(f'test_predictions_super-backbonemodel_{model_idx}.csv', index=False)
        test_predictions_sub.to_csv(f'test_predictions_sub-backbonemodel_{model_idx}.csv', index=False)

    if return_predictions:
        return test_predictions


def plot_graphs(data_list, data_name, hyperparameter_names, num_epochs, is_super=True):
    print('Plot graphs...')
    plot_graph(hyperparameter_names, data_name, data_list, is_super, num_epochs)

    print('Plot graphs done.')


def plot_graph(hyperparameter_names, data_name, data_list, is_super, num_epochs):
    epochs = list(range(1, num_epochs + 1))
    num_hyperparameters = len(hyperparameter_names)
    num_data = len(data_list)

    if not (num_data == num_hyperparameters):
        return print('data list length not the same as hyperparameters')

    plt.figure(figsize=(12, 6))
    for i in range(num_hyperparameters):
        plt.plot(epochs, data_list[i], label=f'{hyperparameter_names[i]}')
    plt.title(f'Training {data_name} vs. Epochs {"Superclass" if is_super else "Subclass"}')
    plt.xlabel('Epochs')
    plt.ylabel(f'{data_name}')
    plt.legend()
    plt.savefig(f'training_{data_name}_{"superclass" if is_super else "subclass"}.png')
    plt.show()


def create_dataloader(train_csv_path, train_data_dir, transform, is_super, batch_size, early_stop, validation_pct,
                      worker_count):
    dataset = AnimalDataset(train_csv_path, train_data_dir, transform=transform, is_super=is_super)

    if early_stop:
        print(f'Using early stop {"superclass" if is_super else "subclass"}...')
        training_dataset, validation_dataset = split_dataset(dataset, validation_pct)
    else:
        training_dataset = dataset
        validation_dataset = None

    dataloader_train = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=worker_count)
    dataloader_validate = None

    if early_stop:
        dataloader_validate = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True,
                                         num_workers=worker_count)

    return dataloader_train, dataloader_validate


def train_and_test_models(save_models, model_idx, wdecay=0.001):
    print(f'current batch size: {batch_size}')
    training_loss_data_tt, training_accuracy_data_tt, validation_loss_data_tt = {'super': [], 'sub': []}, {'super': [],
                                                                                                           'sub': []}, {
                                                                                    'super': [], 'sub': []}

    # Create and initialize the superclass model
    superclass_model = SuperclassModel(num_superclasses + num_novel, use_pretrained, model_idx)
    if use_gpu:
        superclass_model = superclass_model.to('cuda')
    # Create and initialize the subclass model
    subclass_model = SubclassModel(num_subclasses + num_novel, num_superclasses + num_novel, use_pretrained, model_idx)

    if use_gpu:
        subclass_model = subclass_model.to('cuda')
    if superclass_model_path is not None and subclass_model_path is not None:
        print('Loading pretrained models...')
        superclass_model_state_dict = torch.load(superclass_model_path)
        subclass_model_state_dict = torch.load(subclass_model_path)
        superclass_model.load_state_dict(superclass_model_state_dict)
        subclass_model.load_state_dict(subclass_model_state_dict)
        trained_superclass_model = superclass_model
        trained_subclass_model = subclass_model
        save_models = False
    else:
        # Create dataset, augment dataset and create dataloaders
        transform_train = transforms.Compose(
            [transforms.Resize((64, 64)), transforms.RandomVerticalFlip(p=0.3), transforms.RandomHorizontalFlip(p=0.3),
             transforms.RandomRotation(15),
             transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
             transforms.RandomGrayscale(p=0.2), transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), transforms.RandomErasing(p=0.2)])

#        transform_train = transforms.Compose(
#            [transforms.Resize((64, 64)), transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),

        dataloader1_train, dataloader1_validate = create_dataloader(
            train_csv_path, train_data_dir, transform_train, is_super=True,
            batch_size=batch_size, early_stop=early_stop, validation_pct=validation_pct, worker_count=worker_count
        )

        dataloader2_train, dataloader2_validate = create_dataloader(
            train_csv_path, train_data_dir, transform_train, is_super=False,
            batch_size=batch_size, early_stop=early_stop, validation_pct=validation_pct,
            worker_count=worker_count
        )

        start_time = time.time()
        # Train both models

        wdecay_super = wdecay
        wdecay_sub = wdecay

        trained_models, training_loss_data_tt, training_accuracy_data_tt, validation_loss_data_tt = train_models(
            superclass_model, subclass_model,
            dataloader1_train,
            dataloader2_train, epochs_super, epochs_sub,
            lr_super, lr_sub, wdecay_super, wdecay_sub,
            use_gpu, dataloader1_validate,
            dataloader2_validate, plot)
        trained_superclass_model = trained_models[0]
        trained_subclass_model = trained_models[1]
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.4f} seconds")
    # Inference starts (test dataset)
    if not plot:
        transform_test = transforms.Compose(
            [transforms.Resize((64, 64)), transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_dataset = AnimalTestDataset(test_data_dir, transform=transform_test)
        dataloader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=worker_count)
        test(trained_superclass_model, trained_subclass_model, dataloader_test, use_gpu, model_idx, save_to_csv=True)
        print("Inference done.")
    else:
        print('Skipping inference')
    # Save the trained models
    if save_models and not plot:
        super_class_model_file = f'superclass_model-backbonemodel_{model_idx}-epoch_{epochs_super}-early_stop_{early_stop}-weight_decay_{wdecay_super}.pth'
        sub_class_model_file = f'subclass_model-backbonemodel_{model_idx}-epoch_{epochs_super}_{epochs_sub}-early_stop_{early_stop}-weight_decay_{wdecay_sub}.pth'
        torch.save(trained_superclass_model.state_dict(), super_class_model_file)
        torch.save(trained_subclass_model.state_dict(), sub_class_model_file)
        print('Models saved.')
    else:
        print('Models not saved')
    return training_loss_data_tt, training_accuracy_data_tt, validation_loss_data_tt


if __name__ == "__main__":
    # Define parameters
    args = get_args()
    num_superclasses = args.num_superclasses
    num_subclasses = args.num_subclasses
    num_novel = args.num_novel
    epochs_super = args.epochs_super
    epochs_sub = args.epochs_sub
    batch_size = args.batch_size
    backbone_model = args.backbone_model
    lr_super = args.lr_super
    lr_sub = args.lr_sub
    early_stop = args.early_stop
    validation_pct = args.validation_pct
    wdecay_super = args.wdecay_super
    wdecay_sub = args.wdecay_sub
    save_model = args.save_model
    plot = args.plot
    # Check if model paths are provided, set them as None if not
    superclass_model_path = args.superclass_model_path if args.superclass_model_path else None
    subclass_model_path = args.subclass_model_path if args.subclass_model_path else None

    train_data_dir = "./Released_Data/train_shuffle"
    test_data_dir = "./Released_Data/test_shuffle"
    train_csv_path = "./Released_Data/train_data.csv"
    test_csv_path = "./Released_Data/test_data.csv"
    worker_count = 1
    use_gpu = False
    use_pretrained = True

    if torch.cuda.is_available():
        print("GPU is available!")
        use_gpu = True

    # batch_sizes = [16, 32, 64]
    # Add the model you want to run into the dict below: 1: "MobileNet", 2: "SqueezeNet",
    backbone_models = {0: "DenseNet"}
    weight_decays = [0, 0.0001, 0.001]
    hyperparameter_names = []
    for idx, model_name in backbone_models.items():
        hyperparameter_names.append(f'backbone_models={model_name}')
#    for w in weight_decays:
#        hyperparameter_names.append(f'Weight decay={w}')
    super_key = 'super'
    sub_key = 'sub'
    training_loss_data_all, training_accuracy_data_all, validation_loss_data_all = {super_key: [], sub_key: []}, {
        super_key: [], sub_key: []}, {super_key: [], sub_key: []}

    for idx, model_name in backbone_models.items():
#    for wdecay in weight_decays:
        # backbone_model = backbone_models[idx]
        training_loss_data, training_accuracy_data, validation_loss_data = train_and_test_models(save_model, idx)
        training_loss_data_all[super_key].append(training_loss_data[super_key])
        training_loss_data_all[sub_key].append(training_loss_data[sub_key])
        training_accuracy_data_all[super_key].append(training_accuracy_data[super_key])
        training_accuracy_data_all[sub_key].append(training_accuracy_data[sub_key])
        validation_loss_data_all[super_key].append(validation_loss_data[super_key])
        validation_loss_data_all[sub_key].append(validation_loss_data[sub_key])

    all_data = [training_loss_data_all, training_accuracy_data_all, validation_loss_data_all]
    # print(f'ALl data: {all_data}')
    data_names = ['loss', 'accuracy', 'validation_loss']

    if plot:
        print('plotting...')
        for i in range(len(all_data)):
            data_name = data_names[i]
            if data_name == 'validation_loss' and not early_stop:
                continue
            for key, data_list in all_data[i].items():
                if key == super_key:
                    is_super_t = True
                    num_epochs = epochs_super
                else:
                    is_super_t = False
                    num_epochs = epochs_sub
                plot_graphs(data_list, data_name, hyperparameter_names, num_epochs,
                            is_super_t)
