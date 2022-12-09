import numpy as np
import torch
import torchvision
import torchvision.datasets as datasets
import torch.nn.functional as F
import visdom
import torch.nn as nn
import config
from dataset import AddGaussianNoise
import torch.optim as optim
import time
import datetime
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

def train(epoch,train_loader,save_path,batch_size,criterion):
    network.train()
    y_pred_train = []
    y_true_train = []
    train_best_acc = 0.0
    for batch_idx, (data, target) in enumerate(training_dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        y_pred_train.append(output)
        y_true_train.append(target)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(training_dataloader.dataset),
                       100. * batch_idx / len(training_dataloader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * args.batch_size_train) + ((epoch - 1) * len(training_dataloader.dataset)))

            y_pred_train = flatten(y_pred_train)
            y_true_train = flatten(y_true_train)
            train_epoch_acc = metrics.accuracy_score(y_true_train, y_pred_train)
            print('Train set: Accuracy: {:.0f}%'.format(100. * train_epoch_acc))

            if train_epoch_acc > train_best_acc:
                train_best_acc = train_epoch_acc
                torch.save(network.state_dict(), os.path.join(addr, 'model.pth'))
                torch.save(optimizer.state_dict(), os.path.join(addr, 'optimizer.pth'))


def train2(epoch,train_loader,save_path,batch_size,criterion):
    network.train()
    correct = 0.0
    train_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        train_loss += loss.item()
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = 100. * correct / len(train_loader.dataset)
        loss.backward()
        optimizer.step()

        if batch_idx % opt.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * batch_size) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), os.path.join(save_path,'model.pth'))
            torch.save(optimizer.state_dict(),os.path.join(save_path,'optimizer.pth'))

            _, predicted = torch.max(output.data, 1)

    return train_loss, train_acc

def test(test_loader,save_path,best_acc,criterion):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += criterion(output, target)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    test_acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_acc))
    if test_acc > best_acc:
        torch.save(network.state_dict(), os.path.join(save_path,'model_best.pth'))

    return test_loss,test_acc





