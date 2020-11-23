
from data_loader import Train_Dataset
import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
import argparse
from runx.logx import logx
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logx.initialize("./logs/exp9", coolname=True, tensorboard=True)


def train(model, loader, optimizer, scheduler=None):
    print('now LR: ' + str(optimizer.param_groups[0]['lr']))
    model.train()
    total, correct, train_loss = 0.0, 0.0, 0.0
    for step, data in enumerate(loader):
        x, y = data
        x, y = x.to(device), y.to(device)
        out = model(x)
        # print("size:", x.size(), y.size(), out.size())

        loss = nn.CrossEntropyLoss()(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        _, pred = torch.max(out.data, 1)
        total += y.size(0)
        correct += (pred == y).squeeze().sum().cpu().numpy()
        train_loss += loss.item()

        if step % 100 == 0:
            print("step: {0}, loss:{1}".format(step, loss.item()))
    train_acc = correct / total
    train_loss /= step
    logx.msg("train_acc:" + str(train_acc))
    return train_acc, train_loss


def valid(model, loader):
    model.eval()
    total, correct, valid_loss = 0.0, 0.0, 0.0
    with torch.no_grad():
        for step, data in enumerate(loader):
            x, y = data
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = nn.CrossEntropyLoss()(out, y)
            _, pred = torch.max(out.data, 1)
            total += y.size(0)
            correct += (pred == y).squeeze().sum().cpu().numpy()
            valid_loss += loss.item()
    valid_acc = correct / total
    valid_loss /= step
    logx.msg("valid_acc:" + str(valid_acc))
    return valid_acc, valid_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-epochs", type=int, default=30)
    parser.add_argument("-batch_size", type=int, default=4)
    parser.add_argument("-CLR", type=bool, default=False)
    args = parser.parse_args()


    my_model = models.resnext50_32x4d(norm_layer=nn.InstanceNorm2d)
    my_model = my_model.to(device)
    optimizer = optim.Adam(my_model.parameters())
    scheduler = None
    if args.CLR:
        optimizer = optim.SGD(my_model.parameters(), lr=0.1, momentum=0.9)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    normMean = [0.5964188, 0.4566936, 0.3908954]
    normStd = [0.2590655, 0.2314241, 0.2269535]
    train_transformer = transforms.Compose([
        transforms.RandomChoice([
            transforms.ColorJitter(brightness=[0.8, 1.2], contrast=[0.8, 1.2], saturation=[0.8, 1.2], hue=[-0.2, 0.2]),
            transforms.Compose([transforms.Resize((150, 150)), transforms.Resize((200, 200))]),
            transforms.Compose([transforms.RandomResizedCrop(150, scale=(0.78, 1), ratio=(0.90, 1.10), interpolation=2),
                                transforms.Resize((200, 200))]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation((-20, 20), resample=False, expand=False, center=None),
        ]),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomRotation(degrees=(5, 10)),
        transforms.ToTensor(),
        transforms.Normalize(normMean, normStd)
    ])
    valid_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normMean, normStd)
    ])
    train_dataset = Train_Dataset('./data/new_train.csv', './data/train/', transform=train_transformer)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataset = Train_Dataset('./data/new_valid.csv', './data/train/', transform=valid_transformer)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size)
    best_accuracy = 0
    for epoch in range(args.epochs):
        print("epoch:" + str(epoch))
        train_acc, train_loss = train(my_model, train_loader, optimizer, scheduler=scheduler)
        metric_train = {
            'train_acc': train_acc,
            'train_loss': train_loss
        }
        logx.metric('train', metric_train, epoch)
        # torch.save({'state_dict}': my_model.state_dict()}, './weights/resnet50_last.pth')
        valid_acc, valid_loss = valid(my_model, valid_loader)
        metric_valid = {
            'valid_acc': valid_acc,
            'valid_loss': valid_loss
        }
        logx.metric('val', metric_valid, epoch)
        if valid_acc > best_accuracy:
            best_accuracy = valid_acc
            torch.save({'state_dict}': my_model.state_dict()}, './logs/exp9/highest_valid_acc.pth')
        logx.save_model({'state_dict}': my_model.state_dict()}, valid_loss, epoch, higher_better=False,
                        delete_old=True)
        print("current_acc:{0}, best_acc:{1}".format(valid_acc, best_accuracy))
