from torchvision.datasets import VisionDataset
import torchvision.transforms as torch_transforms
from collections import OrderedDict
import torch
from PIL import Image
import os
import json


class ImageNetDataset(VisionDataset):
    
    """
    ImageNet100 Dataset
    """
    def __init__(self, root_dir, label_json_file: str="dataset/Labels.json", 
                 transform=None, target_transform=None, split='train',
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(ImageNetDataset, self).__init__(root_dir, transform=transform,
                                              target_transform=target_transform)
        self.root_dir = root_dir
        self.transform = torch_transforms.ToTensor() if transform is None else transform
        self.target_transform = target_transform
        self.split = split
        self.fine_labels = self._get_fine_labels(label_json_file)
        self.images, self.labels = self._get_image_paths_and_labels(self.root_dir)
        self.num_classes = len(set(self.labels))
        self.device = device

    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
                idx = idx.tolist()
        image, label = self._load_sample(idx)
        return image, label


    def _get_fine_labels(self, json_file):
        labels = OrderedDict()
        with open(json_file) as f:
            label_dict = json.load(f)
            for idx, label in enumerate(label_dict.items()):
                key, value = label
                labels[key] = [idx, value]
        return labels

    
    def _get_image_paths_and_labels(self, root_dir):
        roots = [folder for folder in os.listdir(root_dir) if folder.startswith(self.split)]
        images = []
        labels = []
        for r in roots:
            image_folders = [f for f in os.listdir(os.path.join(root_dir, r))]
            for f in image_folders:        
                for image in os.listdir(os.path.join(root_dir, r, f)):
                    image_path = os.path.join(root_dir, r, f, image)
                    images.append(image_path)
                    labels.append(self.fine_labels[f][0])
        return images, labels
                    


    def _load_sample(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
            image = torch_transforms.ToTensor()(image).to(self.device)
        if self.target_transform:
            label = self.target_transform(label)
            label = torch.tensor(label).to(self.device)
        else:
            label = torch.tensor(label).to(self.device)
        return image, label
