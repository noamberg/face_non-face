import numpy as np
# import pandas as pd
# import torchvision
# from sklearn import metrics
# from sklearn.metrics import classification_report
from torch.utils.data import SubsetRandomSampler
# from comet_ml import Experiment, Optimizer
from balancedBatchSampler import BalancedBatchSampler
from config import Config
import torch
import datetime
import os
# import math
from dataset import MITfaces, MITtrain, MITval
from model import ViT
from utils import mkdir_if_missing, generate_seed, cometml_experiment
import torch.nn as nn
from trainer import validate, train
import timm

args = Config()

if __name__ == '__main__':
    for net_idx, model in enumerate(args.models):
        for th_idx, threshold in enumerate(args.train_sigmoid_threshold):
            # enumerate(opt.get_experiments(project_name=args.project_name,workspace=args.workspace)):
            generate_seed(args.seed)
            # Set the logger
            now = datetime.datetime.now()
            dt_string = now.strftime("%d_%m_%Y____%H_%M_%S")
            print("date and time =", dt_string)

            # CUDA for PyTorch
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda:0" if use_cuda else "cpu")

            # Create log folder with current time and date
            addr = os.path.join(os.getcwd(), 'logs', dt_string)
            mkdir_if_missing(addr)

            # Create the dataset
            dataset = MITfaces(csv_path='./data/train/train2.csv')
            # Creating data indices for training and validation splits:
            dataset_size = len(dataset)
            indices = list(range(dataset_size))
            split = int(np.floor(args.val_split * dataset_size))
            if args.shuffle_dataset:
                np.random.seed(args.seed)
                np.random.shuffle(indices)
            train_indices, val_indices = indices[split:], indices[:split]

            if train_indices.count(6977) == 1:
                train_indices.remove(6977)
            else:
                val_indices.remove(6977)

            # Creating PT data samplers and loaders:
            train_dataset = MITtrain(csv_path='./data/train/train2.csv',train_indices=train_indices)
            val_dataset = MITval(csv_path='./data/train/train2.csv',indices=val_indices)
            # train_sampler = SubsetRandomSampler(train_indices)
            # valid_sampler = SubsetRandomSampler(val_indices)
            # training_dataloader = torch.utils.data.DataLoader(
            #     dataset,
            #     sampler=ImbalancedDatasetSampler(dataset,
            #                                      indices=train_indices,
            #                                      labels=dataset.labels[train_indices],
            #                                      ),
            #     batch_size=args.batch_size,
            # )

            # Create Dataloaders with balanced batch sampler
            training_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                       sampler=BalancedBatchSampler(train_dataset,
                                                                                    labels=train_dataset.labels),
                                                       batch_size=args.batch_size
                                                       )

            validation_dataloder = torch.utils.data.DataLoader(
                val_dataset,
                shuffle=True,
                batch_size=args.batch_size
            )
            # step_factor = len(training_dataloader) // validation_dataloder
            if model == 'ViT':
                network = ViT(
                    image_size=args.image_size,
                    patch_size=args.patch_size,
                    num_classes=1,
                    dim=args.dim, depth=args.depth, heads=args.heads,
                    mlp_dim=args.mlp_dim,
                    dropout=args.dropout,
                    emb_dropout=args.emb_dropout
                ).to(device)
            elif model == 'ResNet18':
                network = timm.create_model('resnet18', pretrained=False, num_classes=1).to(device)
                # network2 = torchvision.models.resnet18(pretrained=True).to(device)
                # network2.fc = nn.Linear(512, 1).to(device)
            elif model == 'ResNet50':
                network = timm.create_model('resnet50', pretrained=False, num_classes=1).to(device)

            optimizer = torch.optim.SGD(network.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.n_epochs * len(training_dataloader)), eta_min=0, last_epoch=-1)
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=452, gamma=0.75)
            criterion = nn.BCEWithLogitsLoss()

            train_losses = []
            train_counter = []
            test_losses = []
            test_counter = [i * len(training_dataloader.dataset) for i in range(args.n_epochs + 1)]
            best_acc = 0.0

            cometml_experiment = cometml_experiment(api_key=args.api_key,
                                                    project_name=args.project_name,
                                                    workspace=args.workspace,
                                                    hyper_params=args)
            cometml_experiment.set_name(f'{model}_{threshold}_{dt_string}')

            # ----------
            #  Training
            # ----------
            best_acc = 0.0
            best_epoch = 0
            for epoch in range(args.n_epochs):
                # Train #
                network.train()
                train_loss, train_acc = train(training_dataloader, network, threshold, criterion, optimizer,
                                              scheduler, epoch, cometml_experiment, addr, device)
                # Validate #
                print('\n-- Validation --\n')
                val_loss, val_accuracy, balanced_acc = validate(validation_dataloder, network, threshold, criterion,
                                          cometml_experiment, epoch, addr, device, best_acc, best_epoch)
                # scheduler.step(val_loss)

                # Save best accuracy #
                if val_accuracy > best_acc:
                    best_acc = val_accuracy
                    best_epoch = epoch

            print('That is it. Best Accuracy: ', best_acc)
            cometml_experiment.end()

        # # Store metrics
        # TNRs.update(metrics_dict['TNR'])
        # TPRs.update(metrics_dict['TPR'])
        # PPVs.update(metrics_dict['PPV'])
        # # Log final curves to comet.ml (PR,roc,TNR_TPR)
        # if epoch == args.n_epochs - 1:
        #     log_curves(TNRs, TPRs, PPVs, epoch, cometml_experiment)
            # Convert metrics to numpy array
            # TNRs = TNRs.prevs
            # TNRs = np.array(flatten(TNRs))
            # TPRs = TPRs.prevs
            # TPRs = np.array(flatten(TPRs))
            # PPVs = PPVs.prevs
            # PPVs = np.array(flatten(PPVs))
            # # Log  pr, roc, TNR_TPR curves to Comet.ml
            # cometml_experiment.log_curve(f"pr-curve", TPRs, PPVs, step=epoch)
            # cometml_experiment.log_curve(f"roc-curve-class", TPRs, PPVs, step=epoch)
            # cometml_experiment.log_curve(f"TNR_TPR-curve-class", TNRs, TPRs, step=epoch)


            # cometml_experiment.log_metric("TNR", TNRs.avg)
            # cometml_experiment.log_metric("TPR", TPRs.avg)
            # cometml_experiment.log_metric("PPV", PPVs.avg)
            # cometml_experiment.log_metric("F1", F1s.avg)

        # if accuracy > best_accuracy:
        #     validation_best_acc = val_acc
        #     torch.save(network.state_dict(), os.path.join(addr, 'model.pth'))
        #     torch.save(optimizer.state_dict(), os.path.join(addr, 'optimizer.pth'))
        # m = nn.Sigmoid()
        # y_pred_train = []
        # y_true_train = []
        # losses = AverageMeter()
        # accuracy = AverageMeter()
        # train_best_acc = 0.0
        # for batch_idx, (data, target) in enumerate(training_dataloader):
        #     data, target = data.to(device), target.to(device)
        #     optimizer.zero_grad()
        #     output = network(data)
        #
        #     target = target.float()
        #     loss = criterion(output, target)
        #     loss.backward()
        #     optimizer.step()
        #     scheduler.step()
        #     losses.update(loss.item())
        #     accuracy.update(acc.item())
        #
        #     y_pred = m(output).cpu().detach().numpy()
        #     y_pred_batch = np.where(y_pred>=0.5,1,0)
        #
        #     if batch_idx > 0:
        #         y_pred_train = np.concatenate((y_pred_train, y_pred_batch), axis=None)
        #         y_true_train = np.concatenate((y_true_train, target.cpu().detach().numpy().astype(int)), axis=None)
        #     else:
        #         y_pred_train = y_pred_batch
        #         y_true_train = target.cpu().detach().numpy().astype(int)
        #
        #     batch_acc = metrics.accuracy_score(target.cpu().detach().numpy().astype(int), y_pred_batch)
        #
        #     if batch_idx % args.log_interval == 0  or batch_idx == (len(training_dataloader)-1):
        #
        #         print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tBatch Accuracy: {:.3f}'.format(
        #             epoch, batch_idx * len(data), len(training_dataloader.sampler),
        #                    100. * batch_idx / len(training_dataloader), loss.item(), batch_acc))
        #         train_losses.append(loss.item())
        #         train_counter.append(
        #             (batch_idx * args.batch_size) + ((epoch - 1) * len(training_dataloader.dataset)))
        #
        #         # Log reuslts to CometML
        #         cometml_experiment.log_metric("train_batch_loss", loss.item(), step=batch_idx)
        #         cometml_experiment.log_metric("train_batch_acc", batch_acc, step=batch_idx)
        #
        #         if batch_idx == len(training_dataloader)-1 :
        #             train_epoch_acc = metrics.accuracy_score(y_true_train, y_pred_train)
        #             print('Train Epoch: {} \tAccuracy: {:.6f}'.format(epoch, train_epoch_acc))
        #             # Log reuslts to CometML
        #             cometml_experiment.log_metric("train_epoch_loss", loss.item(), step=epoch)
        #             cometml_experiment.log_metric("train_batch_acc", batch_acc, step=epoch)
        #
        # #### Validation ####
        # print('\n##### Validation ######\n')
        # val_loss, val_acc = validate(validation_dataloder, network, cometml_experiment, epoch,addr)
        # if val_acc > val_acc_best:
        #     validation_best_acc = val_acc
        #     torch.save(network.state_dict(), os.path.join(addr, 'model.pth'))
        #     torch.save(optimizer.state_dict(), os.path.join(addr, 'optimizer.pth'))

        # network.eval()
        # val_loss = 0.0
        # validation_losses = []
        # validation_counter = []
        # validation_epoch_acc = 0.0
        # y_pred_val = []
        # y_true_val = []
        # with torch.no_grad():
        #     for batch_idx, (data, target) in enumerate(validation_dataloder):
        #         data, target = data.to(device), target.to(device)
        #         output = network(data)
        #         target = target.float()
        #         val_loss = criterion(output, target).item()
        #
        #         y_pred = m(output).cpu().detach().numpy()
        #         y_pred_batch = np.where(y_pred >= 0.5, 1, 0)
        #         if batch_idx > 0:
        #             y_pred_val = np.concatenate((y_pred_val, y_pred_batch), axis=None)
        #             y_true_val = np.concatenate((y_true_val, target.cpu().detach().numpy().astype(int)), axis=None)
        #         else:
        #             y_pred_val = y_pred_batch
        #             y_true_val = target.cpu().detach().numpy().astype(int)
        #
        #         if batch_idx % args.log_interval == 0 or batch_idx == (len(validation_dataloder)-1):
        #             batch_acc = metrics.accuracy_score(target.cpu().detach().numpy().astype(int), y_pred_batch)
        #             print('Validation Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tBatch Accuracy: {:.3f}'.format(
        #                 epoch, batch_idx * len(data), len(validation_dataloder.dataset),
        #                        100. * batch_idx / len(val_dataset), loss.item(), batch_acc))
        #             validation_losses.append(val_loss)
        #             validation_counter.append(
        #                 (batch_idx * args.batch_size) + ((epoch - 1) * len(training_dataloader.dataset)))
        #
        #             # Log reuslts to CometML
        #             cometml_experiment.log_metric("validation_loss", val_loss, step=batch_idx)
        #             cometml_experiment.log_metric("validation_batch_acc", batch_acc, step=batch_idx)
        #
        #             if batch_idx == len(validation_dataloder)-1 :
        #                 validation_epoch_acc = metrics.accuracy_score(y_true_val, y_pred_val)
        #                 print('Validation Epoch: {} \tAccuracy: {:.6f}'.format(epoch, validation_epoch_acc))
        #                 prfscore = metrics.precision_recall_fscore_support(y_true_val, y_pred_val, average='binary')
        #                 # Log reuslts to CometML
        #                 cometml_experiment.log_metric("validation_loss", val_loss, step=batch_idx)
        #                 cometml_experiment.log_metric("validation_batch_acc", batch_acc, step=batch_idx)
        #                 cometml_experiment.log_confusion_matrix(y_true_val, y_pred_val)
        #
        #             if batch_idx == len(validation_dataloder.dataset) - 1:
        #                 validation_epoch_acc = metrics.accuracy_score(y_true_val, y_pred_val)
        #                 print('Validation Epoch: {} \tAccuracy: {:.6f}\n'.format(epoch, validation_epoch_acc))
        #         # if batch_idx % args.log_interval == 0:
        #         #     # y_pred_val = flatten(y_pred_train)
        #         #     # y_true_val = flatten(y_true_train)
        #         #     validation_epoch_acc = metrics.accuracy_score(y_true_train, y_pred_train)
        #         #     # print('Val set: Accuracy: {:.0f}%'.format(100. * train_epoch_acc))
        #         #     print('Val Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         #         epoch, batch_idx * len(data), len(validation_dataloder.dataset),
        #         #                100. * batch_idx / len(validation_dataloder), loss.item()))
        #         #
        #             if validation_epoch_acc > validation_best_acc:
        #                 validation_best_acc = validation_epoch_acc
        #                 torch.save(network.state_dict(), os.path.join(addr, 'model.pth'))
        #                 torch.save(optimizer.state_dict(), os.path.join(addr, 'optimizer.pth'))
        #                 report = classification_report(y_true=y_true_val, y_pred=y_pred_val, output_dict=True)
        #                 df = pd.DataFrame(report).transpose()
        #                 df.to_csv(os.path.join(addr, 'val_report.csv'))
        #
        #
