from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import yaml

class QMULDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.transform = transform
        # Dummy initialization
        self.items = [f"item_{i}.jpg" for i in range(100)] 
    
    def __len__(self):
        return len(self.items)
        
    def __getitem__(self, idx):
        # Dummy Return
        return torch.randn(3, 256, 256), torch.randn(1, 256, 256) # Image, Sketch

class SketchyCOCODataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        # ...
        
    def __len__(self):
        return 100

    def __getitem__(self, idx):
        return torch.randn(3, 512, 512), torch.randn(1, 512, 512)

class PseudosketchesDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        
    def __len__(self):
        return 100
        
    def __getitem__(self, idx):
        return torch.randn(3, 512, 512), torch.randn(1, 512, 512)

def get_dataloader(dataset_name, config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    dataset_cfg = cfg['dataset']
    name = dataset_cfg['name']
    
    transform = transforms.Compose([
        transforms.Resize((dataset_cfg.get('image_size', 256), dataset_cfg.get('image_size', 256))),
        transforms.ToTensor()
    ])
    
    if name == 'QMUL-Sketch+':
        ds = QMULDataset(dataset_cfg['root_dir'], transform=transform)
    elif name == 'SketchyCOCO':
        ds = SketchyCOCODataset(dataset_cfg['root_dir'], transform=transform)
    elif name == 'Pseudosketches':
        ds = PseudosketchesDataset(dataset_cfg['root_dir'], transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {name}")
        
    return DataLoader(ds, 
                      batch_size=dataset_cfg['loader']['batch_size'], 
                      shuffle=dataset_cfg['loader']['shuffle'],
                      num_workers=dataset_cfg['loader']['num_workers'])
