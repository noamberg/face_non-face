import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import Config
import os
import torch
import torch.nn as nn
from utils import AverageMeter, CountMeter, calc_accuracy, convert_to_array, create_dir, plot_roc_curve, \
    plot_pr_curve, calc_metrics, calc_balanced_accuracy, convert_2Dlists_into_2Darray, TNR_TPR_curve


args= Config()
torch.set_printoptions(precision=2, sci_mode=False)

def test(data_loader, model, criterion, device):
    # Reset meters
    losses = AverageMeter()
    accuracy = AverageMeter()
    y_preds = CountMeter()
    y_trues = CountMeter()
    y_preds_th = CountMeter()
    all_embeddings = CountMeter()

    # Set model to evaluation mode
    model.eval()

    # Define sigmoid for final layer
    m = nn.Sigmoid()

    # Save total number of batches and test-set data length
    total_batches = len(data_loader)
    total_data = len(data_loader.sampler)
    print('Total batches: ', total_batches)

    with torch.no_grad():
        # Iterate over data
        for batch_idx, (data, target, filepaths) in tqdm(enumerate(data_loader), total=len(data_loader)):
            # Forward pass
            data = data.to(device)
            target = target.to(device)
            B = data.size(0)
            if model._get_name() == 'ViT':
                logits, embeddings = model(data)
            elif model._get_name() == 'ResNet':
                logits = model(data).squeeze()
                # Save feature vectors for each image
                embeddings = model.forward_features(data).squeeze()

            # Convert embeddings to numpy array
            embeddings = embeddings.cpu().detach().numpy()
            target = target.float()

            # Calculate loss
            loss = criterion(logits, target)

            # Convert predictions to probabilities and threshold them. Convert all to numpy arrays
            y_pred = m(logits).cpu().detach().numpy()
            y_pred_th = np.where(y_pred >= args.test_sigmoid_threshold, 1, 0)
            y_true = target.cpu().detach().numpy().astype(int)

            # Store FP_FN filepaths
            FP_FN_paths, FP_FN = get_FP_FN_filepaths(y_true, y_pred_th, filepaths)

            FP_FN_df = batch_to_csv(FP_FN_paths, y_true[FP_FN], y_pred_th[FP_FN],
                         os.path.join(args.test_dir, 'misclassified_results.csv'), batch_idx)
            copy_FNs_FPs_to_dirs(FP_FN_df, args.test_dir)

            acc = calc_accuracy(y_pred_th, y_true)
            accuracy.update(acc, B)
            losses.update(loss.item(), B)

            all_embeddings.update(embeddings)
            y_preds.update(y_pred)
            y_trues.update(y_true)
            y_preds_th.update(y_pred_th)

            if batch_idx == (total_batches - 1):
                # Convert concatenated y_preds and y_trues lists into one numpy array
                y_preds, y_preds_th, y_trues = \
                    convert_to_array(y_preds.prevs, y_preds_th.prevs, y_trues.prevs)
                # all_embeddings, _, _ = convert_to_array(all_embeddings.prevs, _, _)
                all_embeddings = convert_2Dlists_into_2Darray(all_embeddings.prevs)

                # Create ROC and PR curves for test set
                plots_save_path = create_dir(os.path.join(args.test_dir, 'test_plots'))
                plot_roc_curve(y_trues, y_preds, plots_save_path, epoch=0)
                plot_pr_curve(y_trues, y_preds, plots_save_path, epoch=0)
                TNR_TPR_curve(y_trues, y_preds, plots_save_path, epoch=0)

                # Save  PCA plots, Calculate balanced accuracy and save results to csv
                save_pca_plot(all_embeddings, y_trues, y_preds_th,filepaths,save_path=args.test_dir)
                val_bal_acc = calc_balanced_accuracy(y_trues, y_preds_th)
                print('Balanced Accuracy: {:.3f}'.format(val_bal_acc))

                # Calculate and save confusion matrix
                TNR, TPR, PPV, F1 = calc_metrics(y_trues, y_preds_th,
                                                  args.test_sigmoid_threshold, plots_save_path)
                # Store cm metrics in dictionary
                metrics_dict = {'TNR': TNR, 'TPR': TPR, 'PPV': PPV}

    return accuracy.avg, metrics_dict

# Scatter points on PCA plot and save plot
def pca_plot(projected, indices, save_path, title):
    import matplotlib.pyplot as plt
    import plotly.express as px
    plt.figure()
    # fig = px.scatter(projected, x=0, y=1, color=indices)
    plt.scatter(projected[indices, 0], projected[indices, 1],cmap=plt.cm.get_cmap('coolwarm', 10))  # Greens,
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()
    plt.title('{} PCA points'.format(title))
    plt.savefig(os.path.join(save_path, 'pca_{}.png'.format(title)), dpi=300)

# Save PCA plot
def save_pca_plot(embeddings, y_trues, y_preds, filepaths, save_path):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    projected = pca.fit_transform(embeddings)

    FN_indices = np.where((y_trues != y_preds) & (y_preds == 0))[0]
    FP_indices = np.where((y_trues != y_preds) & (y_preds == 1))[0]
    TP_indices = np.where((y_trues == y_preds) & (y_preds == 1))[0]
    TN_indices = np.where((y_trues == y_preds) & (y_preds == 0))[0]

    pca_plot(projected, FN_indices, save_path, title='FN')
    pca_plot(projected, FP_indices, save_path, title='FP')
    pca_plot(projected, TP_indices, save_path, title='TP')
    pca_plot(projected, TN_indices, save_path, title='TN')

# Save FP and FN images paths of test set to csv
def batch_to_csv(batch_filepaths, y_trues, y_preds, save_path, batch_idx):
    content = {'filepaths': batch_filepaths, 'y_true': y_trues, 'y_pred': y_preds}
    if batch_idx == 0:
        df = pd.DataFrame(content)
        with open(save_path, 'w') as f:
            df.to_csv(f,index=False, line_terminator='\n')
    else:
        df = pd.DataFrame(content)
        with open(save_path, 'a') as f:
            df.to_csv(f, index=False, line_terminator='\n')
    return df

# Get FP and FN images paths of test set
def get_FP_FN_filepaths(y_true, y_pred_th, filepaths):
    FP_FN_paths = []
    FP_FN = np.where(np.any([y_pred_th != y_true], axis=0))[0]
    for i in FP_FN:
        FP_FN_paths.append(filepaths[i])
    return FP_FN_paths, FP_FN

# Copy FP and FN images to FP and FN directories
def copy_FNs_FPs_to_dirs(df, save_dir):
    FP_FN_paths = df['filepaths'].values
    y_true = df['y_true'].values
    y_pred = df['y_pred'].values
    for idx, path in enumerate(FP_FN_paths):
        img = cv2.imread(path)
        if y_true[idx] == 1 and y_pred[idx] == 0:
            if os.path.exists(os.path.join(save_dir, 'FNs')) == False:
                create_dir(os.path.join(save_dir, 'FNs'))
            else:
                resized_img = cv2.resize(img, (224, 224))
                cv2.imwrite(os.path.join(save_dir, 'FNs', os.path.basename(path)), resized_img)
        elif y_true[idx] == 0 and y_pred[idx] == 1:
            if os.path.exists(os.path.join(save_dir, 'FPs')) == False:
                create_dir(os.path.join(save_dir, 'FPs'))
            else:
                resized_img = cv2.resize(img, (224, 224))
                cv2.imwrite(os.path.join(save_dir, 'FPs', os.path.basename(path)), resized_img)


