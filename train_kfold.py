import torch
from runx.logx import logx
from torchvision import models, transforms
import torch.nn as nn
import torch.optim as optim
from data_loader import Train_Dataset
from torch.utils.data import DataLoader
from trans import get_transforms


epochs = 75
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def train(model, loader, optimizer):
    # print('now LR: ' + str(optimizer.param_groups[0]['lr']))
    model.train()
    total, correct, train_loss = 0.0, 0.0, 0.0
    c = nn.CrossEntropyLoss().to(device)
    for step, data in enumerate(loader):
        x, y = data
        x, y = x.to(device), y.to(device)
        out = model(x)
        # print("size:", x.size(), y.size(), out.size())

        loss = c(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, pred = torch.max(out.data, 1)
        total += y.size(0)
        correct += (pred == y).squeeze().sum().cpu().numpy()
        train_loss += loss.item()

        if step % 50 == 0:
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


normMean = [0.5964188, 0.4566936, 0.3908954]
normStd = [0.2590655, 0.2314241, 0.2269535]
# train_transformer = transforms.Compose([
#     # transforms.ColorJitter(brightness=[0.8, 1.2], contrast=[0.8, 1.2], saturation=[0.8, 1.2], hue=[-0.2, 0.2]),
#     transforms.RandomChoice([
#         transforms.Compose([transforms.Resize((150, 150)), transforms.Resize((200, 200))]),
#         transforms.Resize((200, 200))
#     ]),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomRotation((-15, 15), resample=False, expand=False, center=None),
#     transforms.RandomGrayscale(),
#     transforms.ToTensor(),
#     transforms.Normalize(normMean, normStd)
# ])
# valid_transformer = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(normMean, normStd)
# ])

train_transformer, valid_transformer = get_transforms(200)

# for i in range(5):
#     my_model = models.resnext50_32x4d(pretrained=True, zero_init_residual=True)
#     my_model = my_model.to(device)
#     optimizer = optim.Adam(my_model.parameters())
#     logx.initialize("./logs/exp13/fold" + str(i), coolname=True, tensorboard=True)
#     train_dataset = Train_Dataset('./data/train_' + str(i) + '.csv', './data/train', transform=train_transformer)
#     train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
#     valid_dataset = Train_Dataset('./data/valid_' + str(i) + '.csv', './data/train', transform=train_transformer)
#     valid_loader = DataLoader(dataset=valid_dataset, batch_size=64)
#     print('model: ' + str(i) + ' || train_dataset: train_' + str(i) + ' || valid_dataset: valid_' + str(i))
#
#     best_accuracy = 0
#     for epoch in range(epochs):
#         print('model: ' + str(i) + '||epoch: ' + str(epoch))
#         train_acc, train_loss = train(my_model, train_loader, optimizer)
#         metric_train = {
#             'train_acc': train_acc,
#             'train_loss': train_loss
#         }
#         logx.metric('train', metric_train, epoch)
#         valid_acc, valid_loss = valid(my_model, valid_loader)
#         metric_valid = {
#             'valid_acc': valid_acc,
#             'valid_loss': valid_loss
#         }
#         logx.metric('val', metric_valid, epoch)
#         if valid_acc > best_accuracy:
#             best_accuracy = valid_acc
#             torch.save({'state_dict}': my_model.state_dict()}, './logs/exp13/fold' + str(i) + '/highest_valid_acc.pth')
#         logx.save_model({'state_dict}': my_model.state_dict()}, valid_loss, epoch, higher_better=False, delete_old=True)
#         print("current_acc:{0}, best_acc:{1}".format(valid_acc, best_accuracy))
#
#     print('------------------------')

i = 2
my_model = models.resnext50_32x4d(pretrained=False, zero_init_residual=True)
# my_model = models.resnet50(pretrained=False)
my_model = my_model.to(device)
optimizer = optim.Adam(my_model.parameters())
logx.initialize("./logs/exp14/resnext50+da+normlize", coolname=True, tensorboard=True)
train_dataset = Train_Dataset('./data/train_' + str(i) + '.csv', './data/train', transform=train_transformer)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
valid_dataset = Train_Dataset('./data/valid_' + str(i) + '.csv', './data/train', transform=train_transformer)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=64)
print('model: ' + str(i) + ' || train_dataset: train_' + str(i) + ' || valid_dataset: valid_' + str(i))

best_accuracy = 0
for epoch in range(epochs):
    print('model: ' + str(i) + '||epoch: ' + str(epoch))
    train_acc, train_loss = train(my_model, train_loader, optimizer)
    metric_train = {
        'train_acc': train_acc,
        'train_loss': train_loss
    }
    logx.metric('train', metric_train, epoch)
    valid_acc, valid_loss = valid(my_model, valid_loader)
    metric_valid = {
        'valid_acc': valid_acc,
        'valid_loss': valid_loss
    }
    logx.metric('val', metric_valid, epoch)
    if valid_acc > best_accuracy:
        best_accuracy = valid_acc
        torch.save({'state_dict}': my_model.state_dict()}, './logs/exp14/resnext50+da+normlize/highest_valid_acc.pth')
    logx.save_model({'state_dict}': my_model.state_dict()}, valid_loss, epoch, higher_better=False, delete_old=True)
    print("current_acc:{0}, best_acc:{1}".format(valid_acc, best_accuracy))

print('------------------------')
