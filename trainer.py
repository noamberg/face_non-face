import numpy as np
import wandb
from torch import nn
import torch
from tqdm import tqdm
import os
from config import Config
from utils import AverageMeter, CountMeter, calc_accuracy, \
    plot_roc_curve, plot_pr_curve, create_dir, \
    convert_to_array, calc_balanced_accuracy

args = Config()

def validate(data_loader, model, threshold, criterion,epoch, addr, device, best_acc, best_epoch):
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

            step_counter += 1

            if batch_idx % args.log_interval == 0:
                print('\nValidation Epoch: {} [{}/{}]\tLoss: {:.6f}\tBatch Accuracy: {:.3f}'.format(
                    epoch, (batch_idx + 1) * args.batch_size, total_data, loss.item(), acc))

            if batch_idx == (total_batches - 1):

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
                wandb.log({'val/Loss': losses.avg, 'val/Accuracy': accuracy.avg, 'val/Balanced Accuracy': val_bal_acc})

                # Save epoch ROC and PR curves every 10th epoch and at the end of training
                if epoch % 10 == 0 or epoch == args.n_epochs - 1:
                    plots_save_path = create_dir(os.path.join(addr, 'val_plots'))
                    plot_roc_curve(y_trues, y_preds, plots_save_path, epoch)
                    plot_pr_curve(y_trues, y_preds, plots_save_path, epoch)
                    print('Epoch {} Validation Balanced accuracy: {:.3f}'.format(epoch,val_bal_acc))

    print('Loss {loss.avg:.4f}\t'
          'Acc: {acc.avg:.4f} \t'.format(loss=losses, acc=accuracy))

    return losses.avg, accuracy.avg, val_bal_acc

def train(data_loader, model, threshold, criterion, optimizer,scheduler, epoch,addr,device):
    model.train()
    m = nn.Sigmoid()

    if args.pretrained == True:
        model.load_state_dict(torch.load(args.pretrained_model_path))

    losses = AverageMeter()
    accuracy = AverageMeter()
    y_preds_tr = CountMeter()
    y_trues_tr = CountMeter()
    y_preds_thresholded_tr = CountMeter()
    step_counter = 0

    total_batches = len(data_loader)
    total_data = len(data_loader.sampler)
    last_batch = total_data % args.batch_size

    # for param in model.parameters():
    #     param.requires_grad = False
    # for param in model.fc.parameters():
    #     param.requires_grad = True

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

        y_pred = m(output).cpu().detach().numpy()
        y_pred_th = np.where(y_pred >= threshold, 1, 0)
        y_true = target.cpu().detach().numpy().astype(int)

        acc = calc_accuracy(y_pred_th, y_true)

        losses.update(loss.item(), B)
        accuracy.update(acc, B)

        y_preds_tr.update(y_pred)
        y_trues_tr.update(y_true)
        y_preds_thresholded_tr.update(y_pred_th)

        loss.backward()
        optimizer.step()
        scheduler.step()

        # step_counter = epoch * total_batches + batch_idx
        step_counter += 1

        wandb.log({'lr': optimizer.param_groups[0]['lr']}, step=step_counter)

        if batch_idx % args.log_interval == 0:
            print('\nTrain Epoch: {} [{}/{}]\tLoss: {:.6f}\tBatch Accuracy: {:.3f}'.format(
                epoch, (batch_idx + 1) * args.batch_size, total_data, loss.item(), acc))

        if batch_idx == (total_batches - 1):
            # Convert concatenated y_preds and y_trues lists into one numpy array
            y_preds_tr, y_preds_thresholded_tr, y_trues_tr = convert_to_array(y_preds_tr.prevs,
                                                                              y_preds_thresholded_tr.prevs,
                                                                              y_trues_tr.prevs)
            train_bal_acc = calc_balanced_accuracy(y_trues_tr, y_preds_thresholded_tr)
            wandb.log({'train/Loss': losses.avg, 'train/Accuracy': accuracy.avg, 'train/Balanced Accuracy': train_bal_acc})
            print('\nTrain Epoch: {} [{}/{}]\tLoss: {:.6f}\tBatch Accuracy: {:.3f}'.format(
                epoch, batch_idx * args.batch_size + last_batch, total_data, loss.item(), acc))

    # Save optimizer and scheduler states
    torch.save(optimizer.state_dict(), os.path.join(addr, 'optimizer.pth'))
    torch.save(scheduler.state_dict(), os.path.join(addr, 'scheduler.pth'))

    # End of epoch
    print('Epoch {epoch}\t'
          'Loss {loss.avg:.4f}\t'
          'Acc: {acc.avg:.4f} \t'.format(loss=losses, acc=accuracy, epoch=epoch))

    return losses.avg, accuracy.avg