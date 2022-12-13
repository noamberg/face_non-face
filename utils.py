import csv
import glob
import numpy as np
import os
import errno
import torch
import random
from PIL import Image

# Remove the index of the specified element from the list
def remove_last_index(train_indices,val_indices, num):
    if train_indices.count(num) == 1:
        train_indices.remove(num)
    elif val_indices.count(num) == 1:
        val_indices.remove(num)

    return train_indices, val_indices

# Make a directory if it doesn't exist
def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

# Generate seeds for reproducibility
def generate_seed(num):
    random.seed(num)
    np.random.seed(num)
    torch.manual_seed(num)
    torch.cuda.manual_seed(num)

# Create a list of all the files in the directory
def createFileList(myDir, format='.jpg'):
    fileList = []
    print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList

# Convert pgm images to jpg images
def convert_pgm_to_jpg(file_path):
    img = Image.open(file_path)
    img = img.convert("RGB")
    img.save(file_path.replace('.pgm', '.jpg'), "JPEG")

# Append lines to a csv file
def append_line_to_csv(csv_path,filename,filepath,label,height,width,format,mode):
    with open(csv_path, 'a',newline='') as f:
        writer = csv.writer(f)
        writer.writerow([filename,filepath,label,height,width,format,mode])
        f.close()

# Flatten a list of lists
def flatten(t):
    return [item for sublist in t for item in sublist]

# Calculate average of a list
def Average(lst):
    return sum(lst) / len(lst)

# Calculate accuracy of two arrays
def calc_accuracy(preds, labels):
    return (preds == labels).sum().item() / len(labels)

# Accumulate epoch predictions or targets
class CountMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.prevs = []

    def update(self, val, n=1):
        self.val = val
        self.prevs.append(val)

    def __len__(self):
        return len(self.prevs)

# Compute averages
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Save confusion matrix plot
def save_confusion_matrix(cm, save_path, test_sigmoid_threshold):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='g')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(save_path, 'confusion_matrix_test_{}.png'.format(test_sigmoid_threshold)))
    plt.close()

# Create and save TNR TPR curve
def TNR_TPR_curve(y_trues,y_preds,save_path,epoch):
    from sklearn import metrics
    thresholds = np.linspace(0,0.99,num=100)
    import matplotlib.pyplot as plt
    TNR = []
    TPR = []
    for threshold in thresholds:
        y_preds_thresholded = (y_preds > threshold).astype(int)
        cm = metrics.confusion_matrix(y_trues, y_preds_thresholded)
        TN, FP, FN, TP = cm.ravel()
        TNR.append(TN / (TN + FP))
        TPR.append(TP / (TP + FN))

    plt.figure()
    plt.plot(TNR,TPR)
    plt.title('Sensitivity Specificity Curve')
    plt.xlabel('Specificity')
    plt.ylabel('Sensitivity')

    auc = metrics.auc(TNR,TPR)
    plt.legend(['AUC = {:.3f}'.format(auc)])
    plt.savefig(os.path.join(save_path, 'sensitivity_specificity_curve_{}.png'.format(epoch)))
    plt.close()

# Append results to csv file
def append_results_to_csv(train_results_dict, val_results_dict, addr, model, threshold, epoch):
    with open(os.path.join(addr, f'{model}_{threshold}_results.csv'), 'a', newline='') as f:
        writer = csv.writer(f)
        if epoch == 0:
            writer.writerow(['mode', 'epoch', 'loss', 'accuracy',
                             'balanced_acc', 'TNR', 'TPR', 'PPV', 'F1'])
        writer.writerow(train_results_dict.values())
        writer.writerow(val_results_dict.values())

# Calculate confusion matrix and its metrics
def calc_metrics(y_trues, y_preds, sigmoid_threshold, save_path=None):
    # Calculate metrics
    from sklearn import metrics
    confusion_matrix = metrics.confusion_matrix(y_trues, y_preds)
    if save_path is not None:
        save_confusion_matrix(confusion_matrix, save_path, sigmoid_threshold)

    TN, FP, FN, TP = confusion_matrix.ravel()
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)
    # F1 score
    F1 = 2 * PPV * TPR / (PPV + TPR)

    return TNR, TPR, PPV, F1

def log_results_to_csv(save_path, epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, test_sigmoid_threshold, TNR, TPR, PPV, F1):
    with open(os.path.join(save_path, 'results.csv'), 'a',newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, test_sigmoid_threshold, TNR, TPR, PPV, F1])
        f.close()

def plot_roc_curve(y_trues, y_preds, save_path, epoch):
    # Calculate metrics
    from sklearn import metrics
    fpr, tpr, thresholds = metrics.roc_curve(y_trues, y_preds)
    # Plot roc curve
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(fpr, tpr)
    plt.title('ROC Curve')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    auc_roc = metrics.auc(fpr, tpr)
    plt.legend(['AUC = {:.4f}'.format(auc_roc)])
    plt.savefig(os.path.join(save_path, 'roc_curve_{}.png'.format(epoch)))
    plt.close()

# Plot and save the PR curve
def plot_pr_curve(y_trues, y_preds, save_path, epoch):
    # Calculate metrics
    from sklearn import metrics
    precision, recall, thresholds = metrics.precision_recall_curve(y_trues, y_preds)
    # Plot pr curve
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(recall, precision)
    plt.title('PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    auc_pr = metrics.auc(recall, precision)
    plt.legend(['AUC = {:.4f}'.format(auc_pr)])
    plt.savefig(os.path.join(save_path, 'pr_curve_{}.png'.format(epoch)))
    plt.close()

# Create directory
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

# Convert 3 lists of batch outputs to 3 1D numpy outputs arrays
def convert_to_array(y_preds, y_preds_thresholded, y_trues):
    # Convert concatenated lists into 1D to numpy array
    y_preds = np.array(flatten(y_preds))
    y_preds_thresholded = np.array(flatten(y_preds_thresholded))
    y_trues = np.array(flatten(y_trues))

    return y_preds, y_preds_thresholded, y_trues

def convert_2Dlists_into_2Darray(embeddings):
    # Convert 2D list into 2D numpy array
    embeddings = np.vstack(embeddings[:])
    return embeddings

    return y_preds, y_preds_thresholded, y_trues

# Calculate balanced accuracy
def calc_balanced_accuracy(y_trues, y_preds):
    # Calculate metrics
    from sklearn import metrics
    bal_acc = metrics.balanced_accuracy_score(y_trues, y_preds)
    return bal_acc

if __name__ == '__main__':
    # data paths
    train_path = r'C:\Users\Noam\PycharmProjects\Jubaan\data\train\*\*'
    test_path = r'C:\Users\Noam\PycharmProjects\Jubaan\data\test\*\*'

    # Load pgms, convert them to jpgs, save them and create a csv file with the path to the images
    for file in glob.glob(train_path):
        filename, extension = os.path.splitext(file)
        if extension == ".pgm":
            convert_pgm_to_jpg(file)
        print(file)
        img_file = Image.open(file)
        # get original image parameters...
        width, height = img_file.size
        format = img_file.format
        mode = img_file.mode

        train_csv_path = r'C:\Users\Noam\PycharmProjects\Jubaan\data\train\train.csv'
        test_csv_path = r'C:\Users\Noam\PycharmProjects\Jubaan\data\test\test.csv'

        if not os.path.exists(train_csv_path):
            with open(train_csv_path, 'w',newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['filename','filepath','label','height','width','format','mode'])
                f.close()

        append_line_to_csv(train_csv_path,filename=filename.split('\\')[-1],filepath=file,label=filename.split('\\')[-2],
                               height=height,width=width,format=format,mode=mode)



