# -------------------------------------------------------------------
# Inference model for prediction
# -------------------------------------------------------------------

import os

from models.model import Generator
from train import load_checkpoint
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

class InferenceModel(nn.Module):
    """Inference model"""
    def __init__(self, args):
        super(InferenceModel, self).__init__()

        # Device configuration
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            num_gpus = torch.cuda.device_count()
            print(f"Using {num_gpus} GPUs.")
        else:
            self.device = torch.device('cpu')
            num_gpus = 0
            print("Using CPU.")

        # Initialize model (Generator)
        self.generator = Generator(
        img_size=args.model.img_size,
        patch_size=args.model.patch_size,
        in_chans=args.model.in_chans,
        embed_dim=args.model.embed_dim,
        depths=args.model.depths,
        num_heads=args.model.num_heads,
        window_size=args.model.window_size,
        mlp_ratio=args.model.mlp_ratio,
        out_channels=args.model.out_channels).to(self.device)

        self.optimizer = optim.Adam(self.generator.parameters(), lr=args.lr, betas=(0.5, 0.999))

        self.lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer=self.optimizer,
            step_size=args.decay_epochs,
            gamma=args.decay_rate,
        )

        # Load the trained model.
        self.checkpoint_path = os.path.join(args.checkpoint_dir, f'{args.checkpoint_id}_final_model.pth')
        load_checkpoint(self.generator, self.optimizer, self.lr_scheduler, self.checkpoint_path, self.device)
        self.generator.eval()

    def forward(self, input_tensor):
        if not isinstance(input_tensor, torch.Tensor):
            input_tensor = torch.from_numpy(input_tensor) if isinstance(input_tensor, np.ndarray) else torch.tensor(input_tensor)
        input_tensor = input_tensor.to(self.device)
        with torch.no_grad():
            output_tensor = self.generator(input_tensor).squeeze(1)
        return output_tensor.detach().cpu().numpy()

