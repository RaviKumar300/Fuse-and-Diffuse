import argparse
import yaml
import torch
from accelerate import Accelerator
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.diffusion.sdxl_backbone import FuseAndDiffuseSDXL
from data.dataset_factory import get_dataloader
from utils.logger import setup_logger

def train(args):
    accelerator = Accelerator()
    logger = setup_logger("Stage2_Train", save_dir=args.output_dir, distributed_rank=accelerator.process_index)
    logger.info("Starting Stage 2: Diffusion Adapter Training")
    
    # Load Configs
    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)
    
    model = FuseAndDiffuseSDXL(model_config['model'])
    
    # Prepare Dataloader
    dataloader = get_dataloader(args.dataset, args.dataset_config)
    
    model, dataloader = accelerator.prepare(model, dataloader)
    
    global_step = 0
    for epoch in range(args.epochs):
        for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
            with accelerator.accumulate(model):
                # Unpack batch
                # sketch, style, color, prompt ...
                
                # Forward (Conceptual)
                # loss = model(batch)
                
                # Backward
                # accelerator.backward(loss)
                pass
                
    logger.info("Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, default='configs/models/sdxl_adapters.yaml')
    parser.add_argument('--dataset', type=str, default='SketchyCOCO')
    parser.add_argument('--dataset_config', type=str, default='configs/datasets/sketchycoco.yaml')
    parser.add_argument('--output_dir', type=str, default='./work_dirs/stage2')
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    
    train(args)
