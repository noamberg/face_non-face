from utils import AverageMeter, CountMeter, calc_accuracy, \
    plot_roc_curve, plot_pr_curve, create_dir, \
    convert_to_array, calc_balanced_accuracy, calc_metrics, TNR_TPR_curve
import numpy as np
import wandb
from torch import nn
import torch
from tqdm import tqdm
import os
from config import Config

args = Config()

def validate(data_loader, model, threshold, criterion,epoch, addr, device):
    # Reset counters
    losses = AverageMeter()
    accuracy = AverageMeter()
    y_preds = CountMeter()
    y_trues = CountMeter()
    y_preds_thresh = CountMeter()

    # Evaluation mode
    model.eval()
    # Set final layer to Sigmoid
    m = nn.Sigmoid()

    # Save lengths of data loader, sampler and last batch
    total_batches = len(data_loader)
    total_data = len(data_loader.sampler)
    last_batch = total_data % args.batch_size
    step_counter = 0

    # Disable gradient calculation
    with torch.no_grad():
        # Iterate over batches
        for batch_idx, (data, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
            # Foward-pass
            data = data.to(device)
            target = target.to(device)
            B = data.size(0)
            output = model(data)
            target = target.float()

            if model._get_name() == 'ResNet':
                output = output.squeeze()

            # Calculate loss
            loss = criterion(output, target)

            # Convert predictions to probabilities and threshold them. Convert all to numpy arrays
            y_pred = m(output).cpu().detach().numpy()
            y_pred_th = np.where(y_pred >= threshold, 1, 0)
            y_true = target.cpu().detach().numpy().astype(int)

            # Calculate accuracy
            acc = calc_accuracy(y_pred_th, y_true)

            # Update counters
            losses.update(loss.item(), B)
            accuracy.update(acc, B)

            y_preds.update(y_pred)
            y_trues.update(y_true)
            y_preds_thresh.update(y_pred_th)

            step_counter += 1

            if batch_idx % args.log_interval == 0:
                print('\nValidation Epoch: {} [{}/{}]\tLoss: {:.6f}\tBatch Accuracy: {:.3f}'.format(
                    epoch, (batch_idx + 1) * args.batch_size, total_data, loss.item(), acc))

            if batch_idx == (total_batches - 1):
                print('\nValidation Epoch: {} [{}/{}]\tLoss: {:.6f}\tBatch Accuracy: {:.3f}'.format(
                    epoch, batch_idx * args.batch_size + last_batch, total_data, loss.item(), acc))

                # Convert all batches results to 1D numpy arrays
                y_preds, y_preds_thresh, y_trues = \
                    convert_to_array(y_preds.prevs, y_preds_thresh.prevs, y_trues.prevs)

                # Calculate metrics
                val_bal_acc = calc_balanced_accuracy(y_trues, y_preds_thresh)
                val_TNR, val_TPR, val_PPV, val_F1 = calc_metrics(y_trues, y_preds_thresh, args.train_sigmoid_threshold)

                # Log metrics to wandb
                wandb.log({'val/Loss': losses.avg, 'val/Accuracy': accuracy.avg, 'val/Balanced Accuracy': val_bal_acc})

                # Save weights every log_save_interval epochs
                if epoch % args.log_save_interval == 0 or epoch == (args.n_epochs - 1):
                    print('Saving model...')
                    torch.save(model.state_dict(), os.path.join(addr, 'model_epoch{}.pth'.format(epoch)))
                    print('Model saved!')

                # Create and save PR and Sensitivity-Specificity curves
                if epoch % args.log_save_interval == 0 or epoch == args.n_epochs - 1:
                    plots_save_path = create_dir(os.path.join(addr, 'val_plots'))
                    # plot_roc_curve(y_trues, y_preds, plots_save_path, epoch)
                    plot_pr_curve(y_trues, y_preds, plots_save_path, epoch)
                    TNR_TPR_curve(y_trues, y_preds, plots_save_path, epoch)

    print('Loss {loss.avg:.4f}\t'
          'Acc: {acc.avg:.4f} \t'.format(loss=losses, acc=accuracy))

    # Create metrics dictionary
    val_results_dict = {'mode': 'validation', 'epoch': epoch, 'loss': losses.avg, 'acc': accuracy.avg,
                        'val_bal_acc': val_bal_acc, 'val_TNR': val_TNR, 'val_TPR': val_TPR, 'val_PPV': val_PPV,
                        'val_F1': val_F1}

    return val_results_dict

def train(data_loader, model, threshold, criterion, optimizer,scheduler, epoch,addr,device):
    # Training mode (backpropagate and update weights)
    model.train()
    # Set final layer to Sigmoid
    m = nn.Sigmoid()

    # Reset counters
    losses = AverageMeter()
    accuracy = AverageMeter()
    y_preds_tr = CountMeter()
    y_trues_tr = CountMeter()
    y_preds_thresh_tr = CountMeter()
    step_counter = 0

    # Save lengths of data loader, sampler and last batch
    total_batches = len(data_loader)
    total_data = len(data_loader.sampler)
    last_batch = total_data % args.batch_size

    # Iterate over batches
    for batch_idx, (data, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
        # Foward-pass
        data = data.to(device)
        target = target.to(device)
        B = data.size(0)
        optimizer.zero_grad()
        output = model(data)
        target = target.float()

        # Squeeze output Tensor when using ResNet (output shape is (B, 1) instead of (B))
        if model._get_name() == 'ResNet':
            output = output.squeeze()

        # Calculate loss
        loss = criterion(output, target)

        # Convert predictions to probabilities and threshold them. Convert all to numpy arrays
        y_pred = m(output).cpu().detach().numpy()
        y_pred_th = np.where(y_pred >= threshold, 1, 0)
        y_true = target.cpu().detach().numpy().astype(int)

        # Calculate accuracy
        acc = calc_accuracy(y_pred_th, y_true)

        # Update counters
        losses.update(loss.item(), B)
        accuracy.update(acc, B)
        y_preds_tr.update(y_pred)
        y_trues_tr.update(y_true)
        y_preds_thresh_tr.update(y_pred_th)

        # Perform backpropagation and update weights
        loss.backward()
        optimizer.step()
        # scheduler.step()

        step_counter += 1

        if batch_idx % args.log_interval == 0:
            print('\nTrain Epoch: {} [{}/{}]\tLoss: {:.6f}\tBatch Accuracy: {:.3f}'.format(
                epoch, (batch_idx + 1) * args.batch_size, total_data, loss.item(), acc))

        if batch_idx == (total_batches - 1):
            # Convert concatenated y_preds and y_trues lists into one numpy array
            y_preds_tr, y_preds_thresh_tr, y_trues_tr = convert_to_array(y_preds_tr.prevs,
                                                                              y_preds_thresh_tr.prevs,
                                                                              y_trues_tr.prevs)
            # Calculate metrics
            train_bal_acc = calc_balanced_accuracy(y_trues_tr, y_preds_thresh_tr)
            train_TNR, train_TPR, train_PPV, train_F1 = \
                calc_metrics(y_trues_tr, y_preds_thresh_tr, args.train_sigmoid_threshold)

            # Log metrics to wandb
            wandb.log({'train/Loss': losses.avg, 'train/Accuracy': accuracy.avg, 'train/Balanced Accuracy': train_bal_acc})
            wandb.log({'lr': optimizer.param_groups[0]['lr']})

            print('\nTrain Epoch: {} [{}/{}]\tLoss: {:.6f}\tBatch Accuracy: {:.3f}'.format(
                epoch, batch_idx * args.batch_size + last_batch, total_data, loss.item(), acc))

    # Save optimizer and scheduler states
    torch.save(optimizer.state_dict(), os.path.join(addr, 'optimizer.pth'))
    torch.save(scheduler.state_dict(), os.path.join(addr, 'scheduler.pth'))

    # End of epoch
    print('Epoch {epoch}\t'
          'Loss {loss.avg:.4f}\t'
          'Acc: {acc.avg:.4f} \t'.format(loss=losses, acc=accuracy, epoch=epoch))

    # Create train results dictionary
    train_results_dict = {'mode': 'train', 'epoch': epoch, 'loss': losses.avg, 'acc': accuracy.avg,
                          'train_bal_acc': train_bal_acc,'train_TNR': train_TNR, 'train_TPR': train_TPR,
                          'train_PPV': train_PPV, 'train_F1': train_F1 }

    return train_results_dict