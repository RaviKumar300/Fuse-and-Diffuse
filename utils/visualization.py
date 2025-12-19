import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

def visualize_batch(sketches, ground_truth, generated, save_path=None):
    """
    Visualizes a batch of triplets: Sketch -> Ground Truth -> Generated
    """
    sketches = sketches.detach().cpu()
    ground_truth = ground_truth.detach().cpu()
    generated = generated.detach().cpu()

    # Denoormalize if necessary (assuming -1 to 1)
    sketches = (sketches + 1) / 2
    ground_truth = (ground_truth + 1) / 2
    generated = (generated + 1) / 2

    # Concatenate along width
    combined = torch.cat([sketches, ground_truth, generated], dim=3)
    
    grid = make_grid(combined, nrow=4, padding=2)
    grid_np = grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=(15, 15))
    plt.imshow(grid_np)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path)
    # plt.show()
