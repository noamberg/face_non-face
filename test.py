import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.datasets as datasets
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import SubsetRandomSampler
from tqdm import tqdm

import config
import torch.optim as optim
import time
import datetime
from config import Config
import os
import torch
import torch.nn as nn
from dataset import MITval, MITfaces
from model import ViT
from test_utils import test
from utils import AverageMeter, CountMeter, calc_accuracy, convert_to_array, create_dir, plot_roc_curve, plot_pr_curve, \
    balanced_accuracy, calc_metrics

def main():
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print('Using device:', device)

    test_dataset = MITfaces(csv_path='./data/test/test2.csv')
    dataset_size = len(test_dataset)
    indices = list(range(dataset_size))

    if args.shuffle_dataset:
        np.random.seed(args.seed)
        np.random.shuffle(indices)

    test_sampler = SubsetRandomSampler(indices)
    test_dataloder = torch.utils.data.DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=args.test_batch_size
    )

    network = ViT(
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_classes=1,
        dim=args.dim, depth=args.depth, heads=args.heads,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
        emb_dropout=args.emb_dropout,
        mode = 'test'
    ).to(device)

    network2 = torchvision.models.resnet18(pretrained=True).to(device)
    network2.fc = nn.Linear(512, 1).to(device)
    # Load best model
    network2.load_state_dict(torch.load(os.path.join(args.test_dir, 'best_model.pth')))
    # Test the model
    accuracy, metrics_dict = test(test_dataloder, network2, device)

    print('Test Accuracy: {:.4f}'.format(accuracy))
    print('Test TNR: {:.4f}\tTest TPR: {:.4f}\tTest PPV: {:.4f}'.format(metrics_dict['TNR'], metrics_dict['TPR'], metrics_dict['PPV']))

if __name__ == '__main__':
    args = Config()
    main()
