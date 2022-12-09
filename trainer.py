import numpy as np
from sklearn import metrics
from sklearn.metrics import classification_report
from torch import nn
from torch.version import cuda
# from main import criterion
import torch
from tqdm import tqdm
import os
import pandas as pd
from config import Config
from utils import AverageMeter, CountMeter, calc_accuracy, \
    calc_metrics, flatten, log_roc_curve, log_pr_curve, plot_roc_curve, plot_pr_curve, create_dir, \
    convert_to_array, calc_balanced_accuracy

#### Validation ####
# print('\n##### Validation ######\n')
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
#
args = Config()
#
# def log_metrics(res_args,addr,log_interval = 10):
#     epoch, loss, acc, mode, batch_idx, total_batches, total_data, cometml_experiment, y_true_val, y_pred_val = res_args
#     if batch_idx % log_interval == 0 or batch_idx == total_batches:
#         print('Validation Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tBatch Accuracy: {:.3f}'.format(
#             epoch, (batch_idx + 1) * total_data, total_data,
#                    100. * batch_idx / total_data, loss.item(), acc))
#
#     elif batch_idx == total_batches:
#         cometml_experiment.log_metric(mode + '_loss', loss.item(), step=epoch)
#         cometml_experiment.log_metric(mode + '_acc', acc.item(), step=epoch)
#         print('Validation Epoch: {} \tAccuracy: {:.6f}'.format(epoch, validation_epoch_acc))
#         prfscore = metrics.precision_recall_fscore_support(y_true_val, y_pred_val, average='binary')
#         print('Validation Epoch: {} \tPrecision: {:.6f}'.format(epoch, prfscore[0]))
#         print('Validation Epoch: {} \tRecall: {:.6f}'.format(epoch, prfscore[1]))
#         print('Validation Epoch: {} \tF1: {:.6f}'.format(epoch, prfscore[2]))
#         cometml_experiment.log_confusion_matrix(y_true_val, y_pred_val)
#         report = classification_report(y_true=y_true_val, y_pred=y_pred_val, output_dict=True)
#         df = pd.DataFrame(report).transpose()
#         df.to_csv(os.path.join(addr, 'val_report.csv'))


def validate(data_loader, model, threshold, criterion, cometml_experiment,epoch, addr, device, best_acc, best_epoch):
    losses = AverageMeter()
    accuracy = AverageMeter()
    y_preds = CountMeter()
    y_trues = CountMeter()
    y_preds_thresholded = CountMeter()

    model.eval()
    m = nn.Sigmoid()

    total_batches = len(data_loader)
    total_data = len(data_loader.sampler)
    last_batch = total_data % args.batch_size
    step_counter = 0

    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
            data = data.to(device)
            target = target.to(device)
            B = data.size(0)
            output = model(data)
            target = target.float()

            # If we use ResNet18, we need to squeeze the output Tensor
            output = output.squeeze()

            loss = criterion(output, target)

            y_pred = m(output).cpu().detach().numpy()
            y_pred_th = np.where(y_pred >= threshold, 1, 0)
            y_true = target.cpu().detach().numpy().astype(int)

            acc = calc_accuracy(y_pred_th, y_true)

            losses.update(loss.item(), B)
            accuracy.update(acc, B)

            y_preds.update(y_pred)
            y_trues.update(y_true)
            y_preds_thresholded.update(y_pred_th)

            # if batch_idx % log_interval == 0 or batch_idx == total_batches:
            #     print('Validation Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tBatch Accuracy: {:.3f}'.format(
            #         epoch, (batch_idx + 1) * total_data, total_data,
            #                100. * batch_idx / total_data, loss.item(), acc))
            step_counter = epoch * total_batches + batch_idx
            cometml_experiment.log_metric('val_loss_Step', loss.item(), step=step_counter)
            cometml_experiment.log_metric('val_acc_Step', acc, step=step_counter)

            if batch_idx % args.log_interval == 0:
                print('\nValidation Epoch: {} [{}/{}]\tLoss: {:.6f}\tBatch Accuracy: {:.3f}'.format(
                    epoch, (batch_idx + 1) * args.batch_size, total_data, loss.item(), acc))

            if batch_idx == (total_batches - 1):
                cometml_experiment.log_metric('val_loss_Epoch', losses.avg, step=epoch)
                cometml_experiment.log_metric('val_acc_Epoch', accuracy.avg, step=epoch)
                # cometml_experiment.log_confusion_matrix(y_trues, y_preds)

                print('\nValidation Epoch: {} [{}/{}]\tLoss: {:.6f}\tBatch Accuracy: {:.3f}'.format(
                    epoch, batch_idx * args.batch_size + last_batch, total_data, loss.item(), acc))

                if accuracy.avg > best_acc:
                    best_acc = accuracy.avg
                    print('Saving model...')
                    torch.save(model.state_dict(), os.path.join(addr, 'best_model_epoch{}.pth'.format(epoch)))
                    if os.path.exists(os.path.join(addr, 'best_model_epoch{}.pth'.format(best_epoch)))\
                            and epoch != 0:
                        os.remove(os.path.join(addr, 'best_model_epoch{}.pth'.format(best_epoch)))
                    best_epoch = epoch
                    print('Model saved!')

                # Convert concatenated y_preds and y_trues lists into one numpy array
                y_preds, y_preds_thresholded, y_trues = \
                    convert_to_array(y_preds.prevs, y_preds_thresholded.prevs, y_trues.prevs)
                val_bal_acc = calc_balanced_accuracy(y_trues, y_preds_thresholded)
                cometml_experiment.log_metric('val_balanced_acc_Epoch', val_bal_acc, step=epoch)

                #     y_preds.prevs
                # y_preds = np.array(flatten(y_preds))
                # y_trues = y_trues.prevs
                # y_trues = np.array(flatten(y_trues))

                # Save epoch's roc and pr curves
                # Save epoch's roc and pr curves every 10th epoch
                if epoch % 10 == 0 or epoch == args.n_epochs - 1:
                    plots_save_path = create_dir(os.path.join(addr, 'val_plots'))
                    plot_roc_curve(y_trues, y_preds, plots_save_path, epoch)
                    plot_pr_curve(y_trues, y_preds, plots_save_path, epoch)
                    print('Epoch {} Validation Balanced accuracy: {:.3f}'.format(epoch,val_bal_acc))
                # Save epoch's confusion matrix
                # TNR, TPR, PPV, F1 = calc_metrics(y_trues, y_preds)
                # metrics_dict = {'TNR': TNR, 'TPR': TPR, 'PPV': PPV}

                # prfscore = metrics.precision_recall_fscore_support(y_trues, y_preds, average='binary')
                # cometml_experiment.log_metric('val_precision', PPV, step=epoch)
                # cometml_experiment.log_metric('val_recall', TPR, step=epoch)
                # cometml_experiment.log_metric('val_F1score', F1, step=epoch)


    print('Loss {loss.avg:.4f}\t'
          'Acc: {acc.avg:.4f} \t'.format(loss=losses, acc=accuracy))

    return losses.avg, accuracy.avg, val_bal_acc

def train(data_loader, model, threshold, criterion, optimizer,scheduler, epoch, cometml_experiment,addr,device):
    losses = AverageMeter()
    accuracy = AverageMeter()
    y_preds_tr = CountMeter()
    y_trues_tr = CountMeter()
    y_preds_thresholded_tr = CountMeter()

    model.train()
    m = nn.Sigmoid()
    total_batches = len(data_loader)
    total_data = len(data_loader.sampler)
    last_batch = total_data % args.batch_size
    step_counter = 0

    for batch_idx, (data, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
        data = data.to(device)
        target = target.to(device)
        B = data.size(0)
        optimizer.zero_grad()
        output = model(data)
        target = target.float()
        # If we use ResNet18 then we need to squeeze output Tensor
        output = output.squeeze()

        loss = criterion(output, target)

        # We set a threshold of 0.5 for the sigmoid of the output to track output in classification accuracy during train
        y_pred = m(output).cpu().detach().numpy()
        y_pred_th = np.where(y_pred >= threshold, 1, 0)
        y_true = target.cpu().detach().numpy().astype(int)

        acc = calc_accuracy(y_pred, y_true)

        losses.update(loss.item(), B)
        accuracy.update(acc, B)

        y_preds_tr.update(y_pred)
        y_trues_tr.update(y_true)
        y_preds_thresholded_tr.update(y_pred_th)

        loss.backward()
        optimizer.step()
        scheduler.step()

        step_counter = epoch * total_batches + batch_idx
        cometml_experiment.log_metric('train_loss_Step', loss.item(), step=step_counter)
        cometml_experiment.log_metric('train_acc_Step', acc, step=step_counter)
        cometml_experiment.log_metric('lr', optimizer.param_groups[0]['lr'], step=step_counter)

        if batch_idx % args.log_interval == 0:
            print('\nTrain Epoch: {} [{}/{}]\tLoss: {:.6f}\tBatch Accuracy: {:.3f}'.format(
                epoch, (batch_idx + 1) * args.batch_size, total_data, loss.item(), acc))

        if batch_idx == (total_batches - 1):
            cometml_experiment.log_metric('train_loss_Epoch', losses.avg, step=epoch)
            # cometml_experiment.log_metric('train_acc_Epoch', accuracy.avg, step=epoch)

            # Convert concatenated y_preds and y_trues lists into one numpy array
            y_preds_tr, y_preds_thresholded_tr, y_trues_tr = convert_to_array(y_preds_tr.prevs,
                                                                              y_preds_thresholded_tr.prevs,
                                                                              y_trues_tr.prevs)

            train_bal_acc = calc_balanced_accuracy(y_trues_tr, y_preds_thresholded_tr)
            cometml_experiment.log_metric('train_balanced_acc_Epoch', train_bal_acc, step=epoch)

            print('\nTrain Epoch: {} [{}/{}]\tLoss: {:.6f}\tBatch Accuracy: {:.3f}'.format(
                epoch, batch_idx * args.batch_size + last_batch, total_data, loss.item(), acc))

        # if batch_idx % args.log_interval == 0:
        #     print('\nTrain Epoch: {} [{}/{}]\tLoss: {:.6f}\tBatch Accuracy: {:.3f}'.format(
        #         epoch, (batch_idx + 1) * args.batch_size, total_data, loss.item(), acc))
        #
        # if batch_idx == (total_batches - 1):
        #     print('\nTrain Epoch: {} [{}/{}]\tLoss: {:.6f}\tBatch Accuracy: {:.3f}'.format(
        #         epoch, batch_idx * args.batch_size + last_batch, total_data, loss.item(), acc))
        #     cometml_experiment.log_metric('train_loss_Epoch', loss.item(), step=epoch)
        #     cometml_experiment.log_metric('train_acc_Epoch', acc, step=epoch)

    # Save optimizer and scheduler states
    torch.save(optimizer.state_dict(), os.path.join(addr, 'optimizer.pth'))
    torch.save(scheduler.state_dict(), os.path.join(addr, 'scheduler.pth'))

    # End of epoch
    print('Epoch {epoch}\t'
          'Loss {loss.avg:.4f}\t'
          'Acc: {acc.avg:.4f} \t'.format(loss=losses, acc=accuracy, epoch=epoch))

    return losses.avg, accuracy.avg