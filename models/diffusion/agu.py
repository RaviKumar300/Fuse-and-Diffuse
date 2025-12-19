import torch
import torch.nn as nn

class AdaptiveGatingUnit(nn.Module):
    """
    AGU: Dynamic arbitration module to weight adapters based on context and timestep.
    """
    def __init__(self, hidden_dim=128, num_adapters=4):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.text_embed_proj = nn.Linear(768, hidden_dim) # Assuming CLIP text dim
        
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_adapters),
            nn.Sigmoid()
        )
        
    def forward(self, timestep, text_embeddings):
        t_emb = self.time_embed(timestep.view(-1, 1).float())
        # Pooling text embeddings for global context
        txt_emb = self.text_embed_proj(text_embeddings.mean(dim=1))
        
        combined = torch.cat([t_emb, txt_emb], dim=1)
        gates = self.gate_net(combined)
        return gates # [B, 4] -> structure, style, color, text weights
