import torchvision.transforms as torch_transforms
from AlexNet import ImageNetDataset
from AlexNet import AlexNet
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import argparse
import os

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

parser = argparse.ArgumentParser(prog='eval',
                                 description="Evaluates the trained model")

parser.add_argument("-td", "--test-dataset-path", type=str,
                    help="Path to the test dataset",
                    required=True)
parser.add_argument("-bs", "--batch-size", type=int,
                    help="Batch size for test dataset.", required=True)
parser.add_argument("-ch", "--checkpoint-path", type=str,
                    help="Checkpoint path that will be evaluated")

args = parser.parse_args()





def eval(rank, world_size):

    dist.init_process_group(backend='gloo',
                            rank=rank,
                            world_size=world_size)
    

    dataset_path = args.test_dataset_path
    batch_size = args.batch_size
    checkpoint_path = args.checkpoint_path
    test_transform = torch_transforms.Compose([
        torch_transforms.Resize(227),
        torch_transforms.CenterCrop(227)
    ])

    test_dataset = ImageNetDataset(dataset_path,
                                f'{dataset_path}/Labels.json',
                                transform=test_transform,
                                split='val')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    test_model = AlexNet()
    loaded_dict = torch.load(checkpoint_path, map_location=torch.device(rank))
    test_model.load_state_dict(loaded_dict['state_dict'])

    test_model.to(rank)
    test_model = DDP(test_model, device_ids=[rank])
    test_model.eval()


    correct1 = 0
    correct5 = 0
    test_size = len(test_dataset)


    for images, labels in tqdm(test_loader):
        outputs = test_model(images.float().to(rank))
        labels = labels.to(rank)
        _, predicted = torch.max(outputs, 1)
        correct1 += (predicted == labels).type(torch.float).sum().item()
        
        _, top5_pred = outputs.topk(5, 1, largest=True, sorted=True)
        
        correct5 += (labels.unsqueeze(1).expand_as(top5_pred) == top5_pred).sum().item()
        
    top1_accuracy = 100 * correct1 / test_size
    top5_accuracy = 100 * correct5 / test_size

    print('Top-1 Accuracy:', top1_accuracy)
    print('Top-5 Accuracy:', top5_accuracy)


def main():
    world_size = torch.cuda.device_count()
    mp.spawn(eval,
             args=(world_size,),
             nprocs=world_size,
             join=True)
    

if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()