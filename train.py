from AlexNet import ImageNetDataset
from AlexNet import AlexNet

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as torch_transforms
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
from time import time
from collections import OrderedDict
import os

import argparse


parser = argparse.ArgumentParser(prog='train',
                                 description='Trains the AlexNet with \
                                    given model hyperparameters.')

parser.add_argument("-tb", "--train-batch-size", type=int,
                    help="Number of training batches.",
                    required=True)
parser.add_argument("-vb", "--validation-batch-size", type=int,
                    help="Number of validation batches.",
                    required=True)
parser.add_argument("-op", "--optimizer", type=str,
                    help="Optimizer for the training",
                    choices=['adam', 'adamw', 'sgd', 'rmsprop'],
                    required=True)
parser.add_argument("-dp", "--dataset-path", type=str,
                    help="ImageNet dataset path.", required=True)
parser.add_argument("-lr", "--learning-rate", type=float,
                    help="Learning rate", required=True)
parser.add_argument("-e", "--epochs", type=int, help="Number of epochs",
                    required=True)
parser.add_argument("-vs", "--validation-size", type=float,
                    help="Validation size as percentage for instance 0.2",
                    required=True)
parser.add_argument("-j", "--json-label-file", type=str,
                    help="JSON file that contains class labels with names.")
parser.add_argument("-cp", "--checkpoint-path", type=str,
                    help="The path that the checkpoint will be saved")
parser.add_argument("-nc", "--number-of-classes", type=int,
                    help="Number of classes in the output.")
args = parser.parse_args()




num_epochs = args.epochs
learning_rate = float(args.learning_rate)
number_of_classes = args.number_of_classes
model = AlexNet(num_classes=number_of_classes)

if args.optimizer == "adam":
    optimizer = torch.optim.Adam(model.parameters, lr=learning_rate)
elif args.optimizer == "adamw":
    optimizer = torch.optim.AdamW(model.parameters, lr=learning_rate)
elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(model.parameters, lr=learning_rate)
elif args.optimizer == "rmsprop":
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

train_batch_size = args.train_batch_size
val_batch_size = args.validation_batch_size
dataset_path = args.dataset_path
val_size = args.validation_size
json_label_file = args.json_label_file
checkpoint_path = args.checkpoint_path


def train(rank, world_size):

    dist.init_process_group(backend='gloo',
                            rank=rank,
                            world_size=world_size)
    
    
    model.to(rank)    
    model = DDP(model, device_ids=[rank])
    
    criterion = torch.nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                        gamma=.9,
                                                        verbose=True)


    scaler = torch.cuda.amp.GradScaler(enabled=True)

    ds_transform = torch_transforms.Compose([
        torch_transforms.Resize(227),
        torch_transforms.CenterCrop(227)
    ])

    dataset = ImageNetDataset(dataset_path,
                        transform=ds_transform,
                        label_json_file=json_label_file)

    val_size = int(len(dataset) * val_size)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_steps = len(train_ds)
    val_steps = len(val_ds)
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=True)

    history = OrderedDict([('train_loss', []),
                        ('train_acc', []),
                        ('val_loss', []),
                        ('val_acc', [])])    

    start_time = time()

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        train_num_of_correct_preds = 0
        val_num_of_correct_preds = 0
        tqdm_train_loader = tqdm(train_loader)
        tqdm_val_loader = tqdm(val_loader)
        for image_batch, labels in tqdm_train_loader:
            tqdm_train_loader.set_description(f'Epoch (train) {epoch + 1}/{num_epochs}')
            with torch.cuda.amp.autocast():
                image_batch, labels = image_batch.float().to(rank), labels.to(rank)
                outputs = model(image_batch)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            epoch_train_loss += loss.item()
            train_num_of_correct_preds += (outputs.argmax(1) == labels).type(torch.float).sum().item()

        with torch.no_grad():
            tqdm_val_loader.set_description(f'Epoch (val) {epoch + 1}/{num_epochs}')
            model.eval()
            for image_batch, labels in tqdm_val_loader:
                with torch.cuda.amp.autocast():
                    image_batch, labels = image_batch.float().to(rank), labels.to(rank)

                    outputs = model(image_batch)
                    loss = criterion(outputs, labels)
                    epoch_val_loss += loss.item()
                val_num_of_correct_preds += (outputs.argmax(1) == labels).type(torch.float).sum().item()

        avg_train_loss = epoch_train_loss / train_steps
        avg_val_loss = epoch_val_loss / val_steps
        if epoch % 3 == 0:
            lr_scheduler.step()
        train_acc = train_num_of_correct_preds / train_steps
        val_acc = val_num_of_correct_preds / val_steps

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        print(f'Epoch {epoch + 1}/{num_epochs}')

        print(f'Train loss: {avg_train_loss:.4f}, Train acc: {train_acc:.4f}')
        print(f'Val loss: {avg_val_loss:.4f}, Val acc: {val_acc:.4f}')

        if (epoch + 1) % 10 == 0:            
            history['state_dict'] = model.module.state_dict()
            history['optimizer_state_dict'] = optimizer.state_dict()
            history['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
            history['scaler_state_dict'] = scaler.state_dict()
            if not os.path.isdir(checkpoint_path):
                os.mkdir(checkpoint_path)
            torch.save(history, f'{checkpoint_path}/checkpoint_{(epoch + 1) % 10}.pth')


def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train,
             args=(world_size,),
             nprocs=world_size,
             join=True)
    

if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()