import torch
import torch.nn as nn
from transformers import CLIPVisionModelWithProjection

class StyleAdapter(nn.Module):
    """
    IP-Adapter style implementation for style injection.
    """
    def __init__(self, clip_model="laion/CLIP-ViT-H-14-laion2B-s32B-b79K"):
        super().__init__()
        self.clip_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_model)
        self.image_proj = nn.Linear(1024, 4096) # Projection to cross-attention dim
        self.num_tokens = 4

    def forward(self, style_image):
        clip_embeds = self.clip_encoder(style_image).image_embeds
        style_tokens = self.image_proj(clip_embeds).reshape(-1, self.num_tokens, 1024)
        return style_tokens
