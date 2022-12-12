from utils import mkdir_if_missing, generate_seed, remove_last_index, append_results_to_csv
from torch.utils.data import SubsetRandomSampler
from balancedBatchSampler import BalancedBatchSampler
from dataset import MITfaces, MITtrain, MITval
from trainer import validate, train
import numpy as np
import wandb
from config import Config
import torch
import datetime
import os
from model import ViT
import torch.nn as nn
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

            # Set 50%-50% batch sampler and create train dataloader
            training_dataloader = \
                torch.utils.data.DataLoader(train_dataset,
                                            sampler=BalancedBatchSampler(train_dataset, labels=train_dataset.labels),
                                            batch_size=args.batch_size, num_workers=args.num_workers)

            # Create validation dataloader
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
            elif model == 'ResNet50':
                network = timm.create_model('resnet50', pretrained=True, num_classes=1).to(device)


            # Set an optimizer
            optimizer = torch.optim.SGD(network.parameters(),
                                        lr=args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay
                                        )
            # Set a scheduler
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=0.97)

            # Set a loss function
            criterion = nn.BCEWithLogitsLoss()

            # Save the current config file to log folder
            with open(os.path.join(addr, f'{model}_{threshold}.txt'), 'w') as f:
                for key, value in vars(args).items():
                    f.write('%s:%s' % (key, value))

            best_TPR = 0.0
            for epoch in range(args.n_epochs):
                # Train the model
                print('\n-Train-\n')
                train_results_dict = train(training_dataloader,network,threshold, criterion,
                                           optimizer,scheduler,epoch,addr,device)
                # Validate the model
                print('\n-Validation-\n')
                val_results_dict = validate(validation_dataloder, network, threshold, criterion,
                                            epoch, addr, device)

                #Save the best model
                if val_results_dict['val_TPR'] > best_TPR:
                    best_TPR = val_results_dict['val_TPR']
                    best_epoch = epoch
                    torch.save(network.state_dict(), os.path.join(addr, f'Best_{model}_{threshold}.pth'))

                # Append train and validation results to csv log file
                append_results_to_csv(train_results_dict, val_results_dict, addr, model, threshold, epoch)

                # Schedule the learning rate, use warmup for the first 5 epochs
                if epoch >= args.warmup_epochs:
                    scheduler.step()

            run.finish()
            print('Finished to train {} with sigmoid threshold {}\n'.format(model, threshold))
        print('Finished {}'.format(model))


if __name__ == '__main__':
    args = Config()
    main(args)
    print('Finished.')


