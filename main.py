import numpy as np
import wandb
from torch.utils.data import SubsetRandomSampler
from balancedBatchSampler import BalancedBatchSampler
from config import Config
import torch
import datetime
import os
from dataset import MITfaces, MITtrain, MITval
from model import ViT
from utils import mkdir_if_missing, generate_seed, remove_last_index
import torch.nn as nn
from trainer import validate, train
import timm

def main(args):
    # Run loop for models
    for net_idx, model in enumerate(args.models):
        # Run loop for sigmoid thresholds for each model
        for th_idx, threshold in enumerate(args.train_sigmoid_threshold):
            run = wandb.init(project="face_non-faces", entity="noamberg", reinit=True)

            # Generate seeds for reproducibility
            generate_seed(args.seed)

            # Set the logger
            now = datetime.datetime.now()
            dt_string = now.strftime("%d_%m_%Y____%H_%M_%S")
            print("date and time =", dt_string)

            # Set run name
            run_id = str(f'{model}_{threshold}_{dt_string}')
            wandb.run.name = run_id
            wandb.run.save()

            # CUDA for PyTorch
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda:0" if use_cuda else "cpu")

            # Create log folder with current time and date
            addr = os.path.join(os.getcwd(), 'logs', dt_string)
            mkdir_if_missing(addr)

            # Create the dataset from the train csv file
            dataset = MITfaces(csv_path='./data/train/train2.csv')

            # Creating data indices for training and validation splits:
            dataset_size = len(dataset)
            indices = list(range(dataset_size))
            split = int(np.floor(args.val_split * dataset_size))
            if args.shuffle_dataset:
                np.random.seed(args.seed)
                np.random.shuffle(indices)

            train_indices, val_indices = indices[split:], indices[:split]
            train_indices, val_indices = remove_last_index(train_indices, val_indices, num=6977)

            # Creating PT data samplers and loaders:
            train_dataset = MITtrain(csv_path='./data/train/train2.csv', train_indices=train_indices)
            val_dataset = MITval(csv_path='./data/train/train2.csv', indices=val_indices)

            training_dataloader = \
                torch.utils.data.DataLoader(train_dataset,
                                            sampler=BalancedBatchSampler(train_dataset, labels=train_dataset.labels),
                                            batch_size=args.batch_size
                                            )
            validation_dataloder = torch.utils.data.DataLoader(val_dataset, shuffle=True, batch_size=args.batch_size)

            # Create the model
            if model == 'ViT':
                network = ViT(
                    image_size=args.image_size,
                    patch_size=args.patch_size,
                    num_classes=1,
                    dim=args.dim, depth=args.depth, heads=args.heads,
                    mlp_dim=args.mlp_dim,
                    dropout=args.dropout,
                    emb_dropout=args.emb_dropout,
                    pretrained=args.pretrained,
                ).to(device)
            elif model == 'ResNet18':
                network = timm.create_model('resnet18', pretrained=True, num_classes=1).to(device)
                # network2 = torchvision.models.resnet18(pretrained=True).to(device)
                # network2.fc = nn.Linear(512, 1).to(device)
            elif model == 'ResNet50':
                network = timm.create_model('resnet50', pretrained=True, num_classes=1).to(device)

            # Set an optimizer
            optimizer = torch.optim.SGD(network.parameters(),
                                        lr=args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay
                                        )
            # Set a scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=(args.n_epochs * len(training_dataloader)),
                                                                   eta_min=0, last_epoch=-1)
            # Set loss function
            criterion = nn.BCEWithLogitsLoss()

            # ----------
            #  Training
            # ----------
            best_acc = 0.0
            best_epoch = 0
            for epoch in range(args.n_epochs):
                # Train the model
                train_loss, train_acc = train(training_dataloader, network, threshold, criterion, optimizer,
                                              scheduler, epoch, addr, device)

                # Validate the model
                print('\n- Validation -\n')
                val_loss, val_accuracy, val_balanced_acc = validate(validation_dataloder,
                                                                    network, threshold, criterion,
                                                                    epoch, addr, device, best_acc,
                                                                    best_epoch
                                                                    )
                # Create results dictionary
                results = {train_loss, train_acc, val_loss, val_accuracy, val_balanced_acc}

                # Save best validation accuracy
                if val_accuracy > best_acc:
                    best_acc = val_accuracy
                    best_epoch = epoch

            run.finish()
            print('Finished {} with sigmoid threshold {}\n'.format(model, threshold))
            print('Results: {} train_loss, {} train_acc, {} val_loss, {} val_accuracy, {} val_balanced_acc'.
                  format(*results))

        print('Finished {}'.format(model))


if __name__ == '__main__':
    args = Config()
    main(args)
    print('Finished.')

# End of file
