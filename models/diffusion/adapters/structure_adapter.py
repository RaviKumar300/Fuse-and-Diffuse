import torch
import torch.nn as nn

class StructureAdapter(nn.Module):
    """
    ControlNet-style adapter for structural guidance.
    """
    def __init__(self, channels=320, downscale_factor=8):
        super().__init__()
        self.input_hint_block = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.SiLU(), 
            nn.Conv2d(32, 96, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(96, 256, 3, padding=1)
        )
        self.zero_conv = nn.Conv2d(320, 320, 1) # Zero convolution

    def forward(self, sketch_hint, diffusion_features):
        hint = self.input_hint_block(sketch_hint)
        # Add to features
        return self.zero_conv(hint) + diffusion_features
