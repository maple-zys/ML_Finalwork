import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from tensorboard.backend.event_processing import event_accumulator


def read_tfevents(filepath):
    filename = glob(filepath + '/events*')[0]
    ea = event_accumulator.EventAccumulator(filename)
    ea.Reload()

    train_loss = [x.value for x in ea.scalars.Items('train/train_loss')]
    train_accuracy = [x.value for x in ea.scalars.Items('train/train_acc')]
    val_loss = [x.value for x in ea.scalars.Items('val/valid_loss')]
    val_accuracy = [x.value for x in ea.scalars.Items('val/valid_acc')]
    return train_loss, train_accuracy, val_loss, val_accuracy

if __name__ == '__main__':

    train_loss_1, train_accuracy_1, val_loss_1, val_accuracy_1 = read_tfevents('./logs/resnet50')
    train_loss_2, train_accuracy_2, val_loss_2, val_accuracy_2 = read_tfevents('./logs/resnet50+da+normlize')
    train_loss_3, train_accuracy_3, val_loss_3, val_accuracy_3 = read_tfevents('./logs/resnext50+da+normlize')
    train_loss_4, train_accuracy_4, val_loss_4, val_accuracy_4 = read_tfevents('./logs/resnext50+da+normlize+pre')
    x = [a for a in range(75)]
    # plt.plot(x, train_accuracy_1, label='resnet50_train')
    # plt.plot(x, val_accuracy_1, label='resnet50_valid')
    # plt.plot(x, train_accuracy_2, label='resnet50+da_train')
    # plt.plot(x, val_accuracy_2, label='resnet50+da_valid')
    # plt.plot(x, train_accuracy_3, label='resnext50+da_train')
    # plt.plot(x, val_accuracy_3, label='resnext50+da_valid')
    plt.plot(x, train_accuracy_4, label='resnext50+da+pre_train')
    plt.plot(x, val_accuracy_4, label='resnext50+da+pre_valid')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('resnext50+da+pre accuracy')
    # plt.plot(x, val_accuracy_1, label='resnet50')
    # plt.plot(x, val_accuracy_2, label='resnet50+da')
    # plt.plot(x, val_accuracy_3, label='resnext50+da')
    # plt.plot(x, val_accuracy_4, label='resnext50+da+pre')
    # plt.xlabel('epoch')
    # plt.ylabel('accuracy')
    # plt.title('valid accuracy')
    # plt.plot(x, train_loss_1, label='resnet50')
    # plt.plot(x, train_loss_2, label='resnet50+da')
    # plt.plot(x, train_loss_3, label='resnext50+da')
    # plt.plot(x, train_loss_4, label='resnext50+da+pre')
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.title('train loss')
    # plt.plot(x, val_loss_1, label='resnet50')
    # plt.plot(x, val_loss_2, label='resnet50+da')
    # plt.plot(x, val_loss_3, label='resnext50+da')
    # plt.plot(x, val_loss_4, label='resnext50+da+pre')
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.title('valid loss')
    plt.legend(loc='lower right', bbox_to_anchor=(1.0, 0))
    # plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
    plt.savefig('./figure/resnext50+da+pre_accuracy.png')
    plt.show()