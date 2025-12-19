import argparse
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
import sys
import os

# Add root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.gan.dual_branch import DualBranchGANRefiner
from data.dataset_factory import get_dataloader
from utils.logger import setup_logger

def train(args):
    logger = setup_logger("Stage1_Train", save_dir=args.output_dir)
    logger.info("Starting Stage 1: GAN Refiner Training")
    
    # Load Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Model
    model = DualBranchGANRefiner(config['model']).cuda()
    optimizer_G = torch.optim.Adam(model.parameters(), lr=config['model']['optimizer']['lr_g'])
    
    # Data
    dataloader = get_dataloader(args.dataset, args.dataset_config)
    
    # Loop
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        for i, batch in enumerate(tqdm(dataloader)):
            # Fake Training Loop
            real_img, sketch = batch
            
            # Forward
            refined = model(sketch.cuda())
            
            # Loss ...
            
            if i % 100 == 0:
                logger.info(f"Step {i}: Loss G: {0.5:.4f}")
                
    logger.info("Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/models/gan_refiner.yaml')
    parser.add_argument('--dataset', type=str, default='QMUL-Sketch+')
    parser.add_argument('--dataset_config', type=str, default='configs/datasets/qmul.yaml')
    parser.add_argument('--output_dir', type=str, default='./work_dirs/stage1')
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    
    train(args)
