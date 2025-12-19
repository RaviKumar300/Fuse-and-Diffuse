import torch
import torch.nn as nn
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel

from .adapters.structure_adapter import StructureAdapter
from .adapters.style_adapter import StyleAdapter
from .adapters.color_adapter import ColorAdapter
from .agu import AdaptiveGatingUnit

class FuseAndDiffuseSDXL(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Load Frozen SDXL Backbone
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            config['base_model'], 
            torch_dtype=torch.float16, 
            variant="fp16", 
            use_safetensors=True
        )
        self.unet = self.pipe.unet
        self.vae = self.pipe.vae
        self.text_encoder = self.pipe.text_encoder
        
        # Freeze backbone
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
        # Initialize Adapters
        self.structure_adapter = StructureAdapter()
        self.style_adapter = StyleAdapter()
        self.color_adapter = ColorAdapter()
        
        # Initialize AGU
        self.agu = AdaptiveGatingUnit()
        
    def forward(self, refined_sketch, style_image, color_image, text_prompt, timestamps):
        # 1. Get Adapter Features
        struct_feats = self.structure_adapter(refined_sketch, None) # Simplification
        style_feats = self.style_adapter(style_image)
        color_feats = self.color_adapter(color_image)
        
        text_inputs = self.pipe.tokenizer(text_prompt, return_tensors="pt").to(self.device)
        text_feats = self.text_encoder(text_inputs.input_ids)[0]
        
        # 2. Get Gating Weights
        # gates: [alpha_struct, alpha_style, alpha_color, alpha_text]
        gates = self.agu(timestamps, text_feats)
        
        # 3. Fuse Features (Conceptual)
        # In practice, this happens inside the CrossAttention layers via hook injection
        # fused_embeddings = (
        #     gates[:, 0] * struct_feats + 
        #     gates[:, 1] * style_feats + 
        #     gates[:, 2] * color_feats + 
        #     gates[:, 3] * text_feats
        # )
        
        # 4. Diffusion Step
        # noise_pred = self.unet(latents, timestamps, encoder_hidden_states=fused_embeddings).sample
        
        return gates # Return for visualization/debugging during dev
