import os
import sys
import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision

from torch.utils.data import Dataset, DataLoader, BatchSampler, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd
import logging

MY_PATH = r".\sample\sample2"

class nCropTransform:
    """Create 1 or more crops of the same image"""
    def __init__(self, transform, n_views):
        self.n_views = n_views
        self.transform = transform

    def __call__(self, x):
        if self.n_views == 1:
            return self.transform(x)
        elif self.n_views > 1:
            return [self.transform(x) for i in range(self.n_views)]


"""
Implementation borrowed from https://arxiv.org/pdf/2004.11362.pdf
Github: https://github.com/HobbitLong/SupContrast/blob/master/losses.py
"""
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        
        # For case of single view batches
        if len(features.shape) < 3:
            features = torch.unsqueeze(features, dim=1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    
class MultiClassImageDataset(Dataset):
    def __init__(self, ann_df, super_map_df, sub_map_df, img_dir):
        self.ann_df = ann_df
        self.super_map_df = super_map_df
        self.sub_map_df = sub_map_df
        self.img_dir = img_dir

    def __len__(self):
        return len(self.ann_df)

    def __getitem__(self, idx):
        img_name = self.ann_df['image'][idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        super_idx = self.ann_df['superclass_index'][idx]
        super_label = self.super_map_df['class'][super_idx]

        sub_idx = self.ann_df['subclass_index'][idx]
        sub_label = self.sub_map_df['class'][sub_idx]

        return image, super_idx, super_label, sub_idx, sub_label
    
class SplitMultiClassImageDataset(Dataset):
    def __init__(self, dataset, split, transform):
        self.dataset = dataset
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.transform:
            image, super_idx, super_label, sub_idx, sub_label = self.dataset[idx]
            if self.split == 'train':
                image_or_images = self.transform(image)
            elif self.split == 'val':
                image_or_images = self.transform(image)

        return image_or_images, super_idx, super_label, sub_idx, sub_label
    

class MultiClassImageTestDataset(Dataset):
    def __init__(self, super_map_df, sub_map_df, img_dir, transform=None):
        self.super_map_df = super_map_df
        self.sub_map_df = sub_map_df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self): # Count files in img_dir
        return len([fname for fname in os.listdir(self.img_dir)])

    def __getitem__(self, idx):
        img_name = str(idx) + '.jpg'
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, img_name

"""
One contrastive learning tail and one classification tail
"""
class MultiTailModel(nn.Module):
    def __init__(self, model_name, target='superclass', project_out_dim=128, feature_dim=2048, multitail=True):
        super().__init__()
        self.model_name = model_name
        if model_name == 'resnet50':
            self.encoder_network = torchvision.models.resnet50(pretrained=True)
            # Replace classification head
            self.encoder_network.fc = nn.Linear(2048, feature_dim)
        # elif model_name == 'inception_v3':
        #     self.upsample = nn.Upsample(size=75, mode='bilinear')
        #     self.encoder_network = torchvision.models.inception_v3(pretrained=True)
        #     self.encoder_network.AuxLogits = None
        #     logging.info(self.encoder_network)
        #     # Replace classification head
        #     self.encoder_network.fc = nn.Linear(2048, feature_dim)
        if multitail:
            # Got 94 percent accuracy on superclass
            # python supcon_train.py --model_dir=.\sample --log_name=test_superclass --mode=train --model_type=resnet50 --target=superclass --multitail --print_freq_batch=20 --device=cuda --batch_size=64 --epochs=10 --optimizer=SGD --learning_rate=0.005  --momentum=0.05 --temp=0.07 --base_temp=0.07 --feature_dim=128 --encoder_fc_dim=512
            if target == 'superclass':
                self.projection_head = nn.Sequential(
                    nn.Linear(feature_dim, feature_dim),
                    nn.ReLU(),
                    nn.Linear(feature_dim, project_out_dim),
                )
            elif target == "subclass":
                self.projection_head = nn.Sequential(
                    nn.Linear(feature_dim, feature_dim),
                    nn.ReLU(),
                    nn.Linear(feature_dim, project_out_dim),
                )
        else:
            self.projection_head = None
        
        if target == 'superclass':
            self.fc = nn.Linear(feature_dim, 4)
        elif target == 'subclass':
            self.fc = nn.Linear(feature_dim, 88)
        else:
            raise ValueError('target must be superclass or subclass')
            
    
    def forward(self, x):
        # logging.info(x.size())
        # Classification tail
        # if self.model_name == "inception_v3":
        #     x = self.upsample(x)
        encoder_output = self.encoder_network(x)
        encoder_output = F.normalize(encoder_output, p=2, dim=1)
        class_logits = self.fc(encoder_output)
        # Contrastive tail
        if self.projection_head:
            z = self.projection_head(encoder_output)
            z_normalized = F.normalize(z, p=2, dim=1)
            return z_normalized, class_logits
        return None, class_logits
            
class Trainer():
    def __init__(self, opt, model, criterion, optimizer, train_loader, val_loader, test_loader=None, device='cpu'):
        self.opt = opt
        self.model = model.to(device)
        self.contrast_criterion = criterion
        self.classification_criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        running_contrastive_loss = 0.0
        running_classification_loss = 0.0
        for i, data in enumerate(self.train_loader):
            # inputs, super_labels, sub_labels = data[0].to(self.device), data[1].to(self.device), data[3].to(self.device)
            inputs, super_labels, sub_labels = data[0], data[1].to(self.device), data[3].to(self.device)
            
            labels = super_labels if self.opt.target == 'superclass' else sub_labels
            self.optimizer.zero_grad()
            
            if isinstance(inputs, list):# If training with 2 > views
                inputs = torch.cat(inputs, dim=0).to(self.device)
            else:
                inputs = inputs.to(self.device)
                
            feature_vect, class_logits = self.model(inputs)
            # logging.info(f"TRAIN {feature_vect.size()}, {class_logits.size()}, {labels.size()}")
                        
            if opt.multitail:
                # Trying to get the contrastive output feature vect into [batch_size, num_views, feature_dim]
                # for arbitrary number of views
                feature_vect = torch.split(feature_vect, [opt.batch_size for i in range(opt.views)], dim=0)
                feature_vect = torch.cat([feature.unsqueeze(1) for feature in feature_vect], dim=1)
                contrastive_loss = self.contrast_criterion(feature_vect, labels)
                labels = labels.repeat(opt.views)
                classification_loss = self.classification_criterion(class_logits, labels)
                loss = contrastive_loss + classification_loss
            else:
                if opt.views > 1:
                    labels = labels.repeat(opt.views)
                    classification_loss = self.classification_criterion(class_logits, labels)
                else:
                    classification_loss = self.classification_criterion(class_logits, labels)
                loss = classification_loss
            loss.backward()
            self.optimizer.step()
            if i % self.opt.print_freq_batch == 0:
                logging.info(f"Batch: {i}, Total Loss = {loss.item()} Contrastive Loss: {contrastive_loss.item() if opt.multitail else None}, Classification Loss: {classification_loss.item()}")
            running_loss += loss.item()
            running_contrastive_loss += contrastive_loss.item() if opt.multitail else 0.0
            running_classification_loss += classification_loss.item()

        logging.info(f'Training loss: {running_loss/i:.3f}')
        logging.info(f'Training contrastive loss: {running_contrastive_loss/i:.3f}')
        logging.info(f'Training classification loss: {running_classification_loss/i:.3f}')

    def validate_epoch(self):
        # super_correct = 0
        # sub_correct = 0
        correct = 0
        total = 0
        running_loss = 0.0
        running_contrastive_loss = 0.0
        running_classification_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                inputs, super_labels, sub_labels = data[0].to(self.device), data[1].to(self.device), data[3].to(self.device)
                labels = super_labels if self.opt.target == 'superclass' else sub_labels
                feature_vect, class_logits = self.model(inputs)
    
                if opt.multitail:
                    contrastive_loss = self.contrast_criterion(feature_vect, labels)
                    classification_loss = self.classification_criterion(class_logits, labels)
                    loss = contrastive_loss + classification_loss
                else:
                    classification_loss = self.classification_criterion(class_logits, labels)
                    loss = classification_loss
                _, predicted = torch.max(class_logits.data, 1)

                total += labels.size(0)
                
                correct += (predicted == labels).sum().item()
                # sub_correct += (sub_predicted == sub_labels).sum().item()
                running_loss += loss.item()
                running_contrastive_loss += contrastive_loss.item() if opt.multitail else 0.0
                running_classification_loss += classification_loss.item()

        logging.info(f'Validation loss: {running_loss/i:.3f}')
        logging.info(f'Validation contrastive loss: {running_contrastive_loss/i:.3f}')
        logging.info(f'Validation classification loss: {running_classification_loss/i:.3f}')
        logging.info(f'Validation {self.opt.target} acc: {100 * correct / total:.2f} %')

def test(test_loader, model1, model2, device, save_to_csv=False, return_predictions=False):
    if not test_loader:
        raise NotImplementedError('test_loader not specified')

    # Evaluate on test set, in this simple demo no special care is taken for novel/unseen classes
    test_predictions = {'image': [], 'superclass_index': [], 'subclass_index': []}
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, img_name = data[0].to(device), data[1]

            super_outputs, sub_outputs = model(inputs)
            
            _, super_predicted = torch.max(super_outputs.data, 1)
            _, sub_predicted = torch.max(sub_outputs.data, 1)

            test_predictions['image'].append(img_name[0])
            test_predictions['superclass_index'].append(super_predicted.item())
            test_predictions['subclass_index'].append(sub_predicted.item())

    test_predictions = pd.DataFrame(data=test_predictions)

    if save_to_csv:
        test_predictions.to_csv('example_test_predictions.csv', index=False)

    if return_predictions:
        return test_predictions


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    
    parser.add_argument('--dataset_dir', type=str, default='../Released_Data',
                        help='dataset dir containing train_data.csv, test_data.csv, superclass_mapping.csv, subclass_mapping.csv')
    # parser.add_argument('--model_dir', type=str,
    #                     help='where to save models from training or which model to specifically load')
    parser.add_argument('--log_name', type=str,
                        help='name of the log file')
    parser.add_argument('--mode', type=str, default="train",
                        help='train, valid, or test?')
    parser.add_argument('--model_type', type=str, default="resnet50",
                        help='which encoder model?')
    parser.add_argument('--target', type=str, default="superclass",
                        help='Superclass or subclass?')
    parser.add_argument('--multitail', action="store_true",
                        help='Train contrastive and classification tails together?')
    parser.add_argument('--views', type=int, default=0,
                        help='How many data augmentations per image?')
    parser.add_argument('--model1_path', type=str, default="",
                        help='Path for superclass model. Only used when mode==test')
    parser.add_argument('--model2_path', type=str, default="",
                        help='Path for subclass model. Only used when mode==test')
    
    parser.add_argument('--print_freq_batch', type=int, default=1,
                        help='print frequency')
    parser.add_argument('--device', type=str, default="cpu",
                        help='Cuda or CPU?')
    # parser.add_argument('--save_freq', type=int, default=50,
    #                     help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=3,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--learning_rate', type=float, default=0.5,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0,
                        help='momentum')

    # # method
    # parser.add_argument('--method', type=str, default='SupCon',
    #                     choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for contrastive loss function')
    parser.add_argument('--base_temp', type=float, default=0.07,
                        help='temperature for loss function')
    parser.add_argument('--feature_dim', type=int, default=2048,
                        help='feature dim')
    parser.add_argument('--project_out_dim', type=int, default=128,
                        help='projection network output dimension')

    # other setting
    # parser.add_argument('--cosine', action='store_true',
    #                     help='using cosine annealing')
    # parser.add_argument('--warm', action='store_true',
    #                     help='warm-up for large batch training')

    opt = parser.parse_args()
    parse_status = True

    # Make the directory for saving model(s) for the experiment
    if opt.mode == "train":
        # os.makedirs(opt.model_dir, exist_ok=True)
        os.makedirs(MY_PATH, exist_ok=True)
    
    if opt.mode == "train" and opt.multitail and opt.views <= 1:
        logging.error("Multitail training requires at least 2 views")
        parse_status = False
    
    if opt.mode == "test" and (opt.model1_path == "" or opt.model2_path == ""):
        logging.error("Test mode requires model1_path and model2_path")
        parse_status = False
    # if not opt.multitail and opt.views > 1:
    #     logging.error("Can only have 1 view single tail training")
    #     parse_status = False
        
    # iterations = opt.lr_decay_epochs.split(',')
    # opt.lr_decay_epochs = list([])
    # for it in iterations:
    #     opt.lr_decay_epochs.append(int(it))

    # opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
    #     format(opt.method, opt.dataset, opt.model, opt.learning_rate,
    #            opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    # if opt.cosine:
    #     opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    # if opt.batch_size > 256:
    #     opt.warm = True
    # if opt.warm:
    #     opt.model_name = '{}_warm'.format(opt.model_name)
    #     opt.warmup_from = 0.01
    #     opt.warm_epochs = 10
    #     if opt.cosine:
    #         eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
    #         opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
    #                 1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
    #     else:
    #         opt.warmup_to = opt.learning_rate

    # opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    # if not os.path.isdir(opt.tb_folder):
    #     os.makedirs(opt.tb_folder)

    # opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    # if not os.path.isdir(opt.save_folder):
    #     os.makedirs(opt.save_folder)

    return opt, parse_status




if __name__ == "__main__":
    try:
        opt, parse_status = parse_option()
        if parse_status == False:
            logging.error("Argument parsing failed")
            sys.exit(1)
            
            
        if opt.mode == "train":
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setLevel(logging.DEBUG)
            
            model_config = f'model-{opt.target}-model_type_{opt.model_type}-multitail_{opt.multitail}-views_{opt.views}-epoch_{opt.epochs}-batch_size-{opt.batch_size}-optim_{opt.optimizer}-lr_{opt.learning_rate}-weight_decay_{opt.weight_decay}-mom_{opt.momentum}-{opt.device}'
                
            logging.basicConfig(
                level=logging.DEBUG,
                format="%(asctime)s [%(levelname)s] %(message)s",
                handlers=[logging.FileHandler(os.path.join(MY_PATH, model_config + ".log"), mode='w'),
                                    stream_handler]
            )
            
            logging.info("Starting Training")
            logging.info(opt)
            # Create the datasets and dataloaders
            train_ann_df = pd.read_csv(os.path.join(opt.dataset_dir, "train_data.csv"))
            super_map_df = pd.read_csv(os.path.join(opt.dataset_dir, "superclass_mapping.csv"))
            sub_map_df = pd.read_csv(os.path.join(opt.dataset_dir, "subclass_mapping.csv"))

            train_img_dir = os.path.join(opt.dataset_dir, "train_shuffle")
            test_img_dir = os.path.join(opt.dataset_dir, "test_shuffle")

            
            # image_preprocessing = transforms.Compose([
            #     transforms.ToTensor(),
            #     transforms.Normalize(mean=(0), std=(1)),
            # ])
            
            transform = None
            if opt.multitail:
                logging.info(f"Multi-tail training with {opt.views} views per image")
                image_preprocessing =  transforms.Compose([
                    transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0), std=(1)),
                ])
                train_transform = nCropTransform(image_preprocessing, opt.views)
            else:
                if opt.views < 1:
                    logging.info(f"Single tail training with {opt.views} views per image")
                    image_preprocessing =  transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0), std=(1)),
                    ])
                    train_transform = image_preprocessing
                else:
                    logging.info(f"Single tail training with {opt.views} view per image")
                    image_preprocessing =  transforms.Compose([
                        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([
                            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                        ], p=0.8),
                        transforms.RandomGrayscale(p=0.2),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0), std=(1)),
                    ])
                    train_transform = nCropTransform(image_preprocessing, opt.views)
                
            validation_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0), std=(1)),
            ])
            # Create train and val split
            # train_dataset = MultiClassImageDataset(train_ann_df, super_map_df, sub_map_df, train_img_dir, transform=image_preprocessing)
            
            # train_dataset = MultiClassImageDataset(train_ann_df, super_map_df, sub_map_df, train_img_dir, transform=train_transform)
            train_dataset = MultiClassImageDataset(train_ann_df, super_map_df, sub_map_df, train_img_dir)

            proportions = [.8, .2]
            lengths = [int(p * len(train_dataset)) for p in proportions]
            lengths[-1] = len(train_dataset) - sum(lengths[:-1])
            #train_dataset, val_dataset = random_split(train_dataset, [0.9, 0.1])
            # Since I'm using PyTorch 1.1.0, I can't use the above line of code
            train_dataset, val_dataset = random_split(train_dataset, lengths)
            # Adding in validation split's transformation
            # val_dataset.dataset.transform = validation_transform
            train_dataset = SplitMultiClassImageDataset(train_dataset, 'train', transform=train_transform)
            val_dataset = SplitMultiClassImageDataset(val_dataset, 'val', transform=validation_transform)
            
            # Create data loaders
            logging.info("Creating DataLoaders")
            logging.info(f"Batch size: {opt.batch_size}, Effective Batch size: {opt.batch_size * opt.views if opt.views > 0 else 1}")
            train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=opt.num_workers)
            val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

            # Create model
            model = MultiTailModel(opt.model_type, target=opt.target, 
                                feature_dim=opt.feature_dim, 
                                project_out_dim=opt.project_out_dim, multitail=opt.multitail).to(opt.device)
            
            model_file = os.path.join(opt.model_dir, model_config + "_" + ".pth")
            logging.info(f"Saving model to {model_file}")
            torch.save(model.state_dict(), model_file)
            sys.exit()
            # Create criterion
            if opt.multitail:
                criterion = SupConLoss(temperature=opt.temp, base_temperature=opt.base_temp)
            else:
                criterion = None

            # Create optimizer
            if opt.optimizer == 'SGD':
                optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate, 
                                    momentum=opt.momentum, weight_decay=opt.weight_decay)
            elif opt.optimizer == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate,
                                        weight_decay=opt.weight_decay)

            # Create trainer
            trainer = Trainer(opt, model, criterion, optimizer, train_loader, val_loader, device=opt.device)

            # Train
            for epoch in range(opt.epochs):
                logging.info(f'Epoch: {epoch}')
                trainer.train_epoch()
                trainer.validate_epoch()
            logging.info("Finished Training. Saving model...")
            
            model_file = os.path.join(opt.model_dir, model_config + "_" + ".pth")
            logging.info(f"Saving model to {model_file}")
            torch.save(model.state_dict(), model_file)

        
        if opt.mode == "test":
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0), std=(1)),
            ])
            # Create test dataset
            test_dataset = MultiClassImageTestDataset(super_map_df, sub_map_df, test_img_dir, transform=test_transform)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            
            model1 = MultiTailModel("superclass", target=opt.target,
                                feature_dim=opt.feature_dim, 
                                project_out_dim=opt.project_out_dim, multitail=opt.multitail).to(opt.device)
            model2 = MultiTailModel("subclass", target=opt.target,
                                feature_dim=opt.feature_dim, 
                                project_out_dim=opt.project_out_dim, multitail=opt.multitail).to(opt.device)
            
            # # Test
            # test_predictions = trainer.test(save_to_csv=True, return_predictions=True)
            # print(test_predictions.head())
    except Exception as e:
        logging.error(e, exc_info=True)
        raise