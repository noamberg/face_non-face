import numpy as np
import pandas as pd
from tqdm import tqdm
from config import Config
import os
import torch
import torch.nn as nn
from utils import AverageMeter, CountMeter, calc_accuracy, convert_to_array, create_dir, plot_roc_curve, \
    plot_pr_curve, calc_metrics, calc_balanced_accuracy, convert_2Dlists_into_2Darray

args= Config()
torch.set_printoptions(precision=2, sci_mode=False)

def test(data_loader, model, device):
    accuracy = AverageMeter()
    y_preds = CountMeter()
    y_trues = CountMeter()
    y_preds_th = CountMeter()
    all_embeddings = CountMeter()

    model.eval()
    m = nn.Sigmoid()

    total_batches = len(data_loader)
    total_data = len(data_loader.sampler)
    print('Total batches: ', total_batches)

    with torch.no_grad():
        for batch_idx, (data, target, filepaths) in tqdm(enumerate(data_loader), total=len(data_loader)):
            data = data.to(device)
            target = target.to(device)
            B = data.size(0)
            if model._get_name() == 'ViT':
                logits, embeddings = model(data)
            elif model._get_name() == 'ResNet':
                logits = model(data).squeeze()
                model2 = nn.Sequential(*list(model.children())[:-1])
                embeddings = model2(data).squeeze()

            embeddings = embeddings.cpu().detach().numpy()
            target = target.float()

            y_pred = m(logits).cpu().detach().numpy()
            y_pred_th = np.where(y_pred >= args.test_sigmoid_threshold, 1, 0)
            y_true = target.cpu().detach().numpy().astype(int)

            FP_FN_paths, FP_FN = get_FP_FN_filepaths(y_true, y_pred_th, filepaths)
            batch_to_csv(FP_FN_paths, y_true[FP_FN], y_pred_th[FP_FN],
                         os.path.join(args.test_dir, 'misclassified_results.csv'), batch_idx)

            acc = calc_accuracy(y_pred_th, y_true)
            accuracy.update(acc, B)

            all_embeddings.update(embeddings)
            y_preds.update(y_pred)
            y_trues.update(y_true)
            y_preds_th.update(y_pred_th)

            # if batch_idx % args.log_interval == 0:
            #     print('\nTest Batch: [{}/{}]\tBatch Accuracy: {:.3f}'.format(
            #         (batch_idx + 1) * args.batch_size, total_data, acc))

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

                # Save t-SNE and PCA plots, Calcuate balanced accuracy and save results to csv
                save_pca_plot(all_embeddings, y_trues, save_path=args.test_dir)
                # save_tsne_plot(all_embeddings, y_trues, save_path=args.test_dir)
                val_bal_acc = calc_balanced_accuracy(y_trues, y_preds_th)
                print('Balanced Accuracy: {:.3f}'.format(val_bal_acc))

                # Save confusion matrix
                TNR, TPR, PPV, F1 = calc_metrics(y_trues, y_preds_th,
                                                 plots_save_path, args.test_sigmoid_threshold)
                metrics_dict = {'TNR': TNR, 'TPR': TPR, 'PPV': PPV}

    return accuracy.avg, metrics_dict

# Save TSNE plot
def save_tsne_plot(embeddings, labels, save_path):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    tsne = TSNE(n_components=2, random_state=0)
    embeddings_2d = tsne.fit_transform(embeddings)
    df = pd.DataFrame({'x': embeddings_2d[:, 0], 'y': embeddings_2d[:, 1], 'label': labels})
    sns.scatterplot(x="x", y="y", hue="label", data=df)
    plt.savefig(os.path.join(save_path, 'tsne_plot.png'))

# Save PCA plot
def save_pca_plot(embeddings, y_trues, save_path):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    projected = pca.fit_transform(embeddings)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(projected[:, 0], projected[:, 1],
                c=y_trues, edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('Greens', 10))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()
    plt.savefig(os.path.join(save_path, 'pca_plot.png'), dpi=300)

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

# Get FP and FN images paths of test set
def get_FP_FN_filepaths(y_true, y_pred_th, filepaths):
    FP_FN_paths = []
    FP_FN = np.where(np.any([y_pred_th != y_true], axis=0))[0]
    for i in FP_FN:
        FP_FN_paths.append(filepaths[i])
    return FP_FN_paths, FP_FN
