import os
import yaml
import time
import torch
import Helper
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms,models
from torch.utils.data import DataLoader
import numpy as np
import numpy
import timm
from timm import create_model
import sys
import argparse
import random
import zipfile
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
#----------------------------------Ensemble

torch.manual_seed(42)

efficient_net = models.efficientnet_b0(pretrained=True)
classifier = nn.Sequential(
  nn.Linear(in_features=1280, out_features=512),
  nn.ReLU(),
  nn.Dropout(p=0.4),
  nn.Linear(in_features=512, out_features=4),
  nn.LogSoftmax(dim=1)  
)
efficient_net.classifier[1]=classifier

resnet = models.resnet18(pretrained=True)
classifier = nn.Sequential(
  nn.Linear(in_features=512, out_features=256),
  nn.ReLU(),
  nn.Dropout(p=0.4),
  nn.Linear(in_features=256, out_features=4),
  nn.LogSoftmax(dim=1)  
)  
resnet.fc = classifier


efficient_net.to("cuda")
resnet.to("cuda")

class Hairloss(torch.utils.data.Dataset): 
    def __init__(self, path, train, validation_split=0.2, seed=42):
        self.path = path
        self.train = train
        self.labels = ['level0', 'level1', 'level2', 'level3']
        if self.train==1:
            transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomVerticalFlip(p=0.5),
                        transforms.Lambda(lambda x: x.rotate(90)),
                        transforms.RandomRotation(10),
                        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.0011, 0.0010, 0.0010], std=[0.229, 0.224, 0.225])
                    ])
        if self.train ==2 or self.train==3:
            transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.0011, 0.0010, 0.0010], std=[0.229, 0.224, 0.225])
                    ])

        if self.train == 1 or self.train == 3:
            fpath = self.path + '/train'
        elif self.train == 2:
            fpath = self.path + '/test'

        all_data = datasets.ImageFolder(fpath, transform) 

        if self.train == 1 or self.train == 3:
            
            num_total = len(all_data)
            indices = list(range(num_total))

            sss = StratifiedShuffleSplit(n_splits=1, test_size=validation_split, random_state=seed)
            train_index, val_index = next(sss.split(indices, [self.labels[label] for _, label in all_data.samples]))

            if self.train == 1:
                self.data = torch.utils.data.Subset(all_data, train_index)
            else:
                self.data = torch.utils.data.Subset(all_data, val_index)

        else:
            self.data = all_data

    def __getitem__(self, index):
        img, target = self.data[index]
        target = int(target)
        return img, target

    def __len__(self):
        return len(self.data)
    

def get_data(args):
    dataset_train = Hairloss(args.data_path, 1, validation_split=0.2, seed=args.seed)
    dataset_val = Hairloss(args.data_path, 3, validation_split=0.2, seed=args.seed)
    dataset_test = Hairloss(args.data_path, 2)  # 테스트 데이터셋 추가

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    dataloader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    return dataloader_train, dataloader_val, dataloader_test


class module():
    def __init__(self, model,epoch,arg):
        self.args = args
        self.model = model
        self.epoch = epoch
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        for n,p in self.model.named_parameters():
            if "classifier" in n or "fc" in n or "head" in n or "fc1" in n:
                p.requires_grad=True
            else:
                p.requires_grad=False


    def train(self, train_loader,val_loader):
        print("Begin Training")

        self.best_val_loss = float('inf')

        start_time = time.time()
        for epoch in range(self.epoch):
            running_loss = self.train_one_epoch(epoch, train_loader)
            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")
             # Validation after each training epoch
            val_loss, val_acc = self.validate(val_loader)
            print(f"Epoch {epoch + 1}, Validation Loss: {val_loss},Validation Acc: {val_acc}")

            # Check if the current validation loss is the best
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                print("Saving the best model...")
                torch.save(self.model.state_dict(), self.args.model_path)
                best_model_path = self.args.model_path

        end_time = time.time()
        print("Training Done")
        elapsed_time = end_time - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"경과 시간: {int(hours)}시간 {int(minutes)}분 {int(seconds)}초")
        return best_model_path

    def train_one_epoch(self,epoch, train_loader):
        running_loss = 0.0
        self.model.train()
        for batch_idx, data in enumerate(train_loader):
            images, labels = data[0], data[1]
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # _, predicted = torch.max(outputs, 1)
            # accuracy = torch.sum(labels == predicted)/len(predicted)
            # print(f"acc: {accuracy}")
            
            loss.backward()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}: {batch_idx}/{len(train_loader)} - Loss: {loss.item()}")
            self.optimizer.step()
            running_loss += loss.item()
        self.optimizer.step()

        return running_loss
    
    def validate(self, val_loader):
            self.model.eval()
            val_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(val_loader):
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    correct_predictions += torch.sum(predicted == labels).item()
                    total_samples += labels.size(0)


            val_acc = correct_predictions / total_samples
            val_loss /= len(val_loader)
            return val_loss , val_acc
        

    def test(self, test_loader):
        print("Begin Test")
        correct_predictions = 0
        total_samples = 0
        if os.path.exists(self.args.model_path):
            self.model.load_state_dict(torch.load(self.args.model_path))
        self.model.eval()
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += torch.sum(predicted == labels).item()
            total_samples += labels.size(0)
        print(f"Test Done")
        accuracy = correct_predictions / total_samples   
        print(f"Test Accuracy: {accuracy}")
        
class MyEnsemble(nn.Module):

    def __init__(self, modelA, modelB, input):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.fc1 = nn.Linear(input, 4)

    def forward(self, x):
        out1 = self.modelA(x)
        out2 = self.modelB(x)
        out = out1 + out2 
        x = self.fc1(out)
        return torch.softmax(x, dim=1)
    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser('scalp')
    parser.add_argument('--batch-size', default=128, type=int, help='Batch size per device')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--data_path', default='/data/alan0408/Hair/dataset', type=str)
    parser.add_argument('--model_path', default='./model', type=str)
    args = parser.parse_args()

def prerun(args,efficient_net,resnet):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device(args.device)    
    dataloader_train, dataloader_val, dataloader_test = get_data(args) 

    prepar=module(efficient_net,10,args)
    best_efficient_path=prepar.train(dataloader_train,dataloader_val)
    print("efficient_net 결과")
    prepar.test(dataloader_test) 
    efficient_net.load_state_dict(torch.load(best_efficient_path))

    prepar=module(resnet,300,args)
    best_resnet_path=prepar.train(dataloader_train,dataloader_val)
    print("resnet 결과")
    prepar.test(dataloader_test)
    resnet.load_state_dict(torch.load(best_resnet_path))

    ensemble = MyEnsemble(resnet, efficient_net, 4)
    ensemble.to(args.device)
    return ensemble


def run(args,ensemble):
    mymodule = module(ensemble,args.epochs,args)
    dataloader_train, dataloader_val, dataloader_test = get_data(args) 
    mymodule.train(dataloader_train,dataloader_val)
    print("앙상블 모델 결과")
    mymodule.test(dataloader_test)


ensemble=prerun(args,efficient_net,resnet)
run(args,ensemble)