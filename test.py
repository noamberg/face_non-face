import numpy as np
import timm
import torchvision
from torch.utils.data import SubsetRandomSampler
from config import Config
import os
import torch
import torch.nn as nn
from dataset import MITval, MITfaces
from model import ViT
from test_utils import test

def main():
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print('Using device:', device)

    # Create dataset from test csv file
    test_dataset = MITfaces(csv_path='./data/test/test2.csv')
    dataset_size = len(test_dataset)
    indices = list(range(dataset_size))

    # Shuffle indices
    if args.shuffle_dataset:
        np.random.seed(args.random_seed)
        np.random.shuffle(indices)

    # Set sampler and dataloader
    test_sampler = SubsetRandomSampler(indices)
    test_dataloder = torch.utils.data.DataLoader(test_dataset, sampler=test_sampler, batch_size=args.test_batch_size)

    # Set model
    if args.test_model == 'ViT':
        network = ViT(
            image_size=args.image_size,
            patch_size=args.patch_size,
            num_classes=1,
            dim=args.dim, depth=args.depth, heads=args.heads,
            mlp_dim=args.mlp_dim,
            dropout=args.dropout,
            emb_dropout=args.emb_dropout,
            mode='test'
        ).to(device)
    elif args.test_model == 'ResNet18':
        network = timm.create_model('resnet18', pretrained=False, num_classes=1).to(device)
    elif args.test_model == 'ResNet50':
        network = timm.create_model('resnet50', pretrained=False, num_classes=1).to(device)

    # Set loss function
    criterion = nn.BCEWithLogitsLoss()

    # Load best model
    network.load_state_dict(torch.load(os.path.join(args.test_dir, args.test_best_model)))

    # Test the model
    accuracy, metrics_dict = test(test_dataloder, network, criterion, device)

    # Print the results
    print('Test Accuracy: {:.4f}'.format(accuracy))
    print('Test TNR: {:.4f}\tTest TPR: {:.4f}\tTest PPV: {:.4f}'.format(metrics_dict['TNR'], metrics_dict['TPR'], metrics_dict['PPV']))

if __name__ == '__main__':
    args = Config()
    main()
