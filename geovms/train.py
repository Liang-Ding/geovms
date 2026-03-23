from models.model import Generator
from dataloaders.dataloaders import create_dataloader

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import os


def save_checkpoint(model, optimizer, scheduler, epoch, train_losses, val_losses, file_path):
    """ Save model checkpoint """
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, file_path)
    print(f"Model checkpoint saved at {file_path}")


def load_checkpoint(model, optimizer, scheduler, file_path, device):
    """ Load model checkpoint including training and validation losses """
    checkpoint = torch.load(file_path, map_location=torch.device(device), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    print(f"Model checkpoint loaded from {file_path}")
    return epoch, train_losses, val_losses


def train(args):
    """Training and validation"""
    args.lr = float(args.lr)
    args.decay_epochs = int(args.decay_epochs)
    args.decay_rate = float(args.decay_rate)

    # Device configuration
    # Check if distributed training is initialized
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Distributed training setup
        distributed = True
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        # Initialize the distributed environment.
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        # Single GPU setup
        distributed = False
        rank = 0
        world_size = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Using device:", device, "| Rank:", rank, "of", world_size)

    # Create dataloaders
    if distributed:
        dataloaders = {
            'train': create_dataloader(data_path=args.dataset.train_data_path,
                                       batch_size=args.batch_size,
                                       num_workers=args.num_workers,
                                       distributed=True,
                                       world_size=world_size,
                                       rank=rank),
            'val': create_dataloader(data_path=args.dataset.val_data_path,
                                     batch_size=args.batch_size,
                                     num_workers=args.num_workers,
                                     distributed=True,
                                     world_size=world_size,
                                     rank=rank)
        }
    else:
        dataloaders = {
            'train': create_dataloader(data_path=args.dataset.train_data_path,
                                       batch_size=args.batch_size,
                                       num_workers=args.num_workers),
            'val': create_dataloader(data_path=args.dataset.val_data_path,
                                     batch_size=args.batch_size,
                                     num_workers=args.num_workers)
        }

    # Initialize model (Generator)
    generator = Generator(
        img_size=args.model.img_size,
        patch_size=args.model.patch_size,
        in_chans=args.model.in_chans,
        embed_dim=args.model.embed_dim,
        depths=args.model.depths,
        num_heads=args.model.num_heads,
        window_size=args.model.window_size,
        mlp_ratio=args.model.mlp_ratio,
        out_channels=args.model.out_channels
    ).to(device)

    # Wrap the model with DDP if in distributed mode
    if distributed:
        nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[local_rank],
            output_device=local_rank
        )

    # loss, optimizer, and LR scheduler
    loss_func = nn.BCELoss()
    optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=args.decay_epochs,
        gamma=args.decay_rate,
    )

    # Load checkpoint if using pretrained
    if args.use_pretrained:
        checkpoint_filepath = os.path.join(args.checkpoint_dir, f'{args.checkpoint_id}_final_model.pth')
        start_epoch, train_losses, val_losses = load_checkpoint(generator, optimizer, lr_scheduler, checkpoint_filepath, device)
        print(f"Resuming training from epoch {start_epoch}.")
    else:
        start_epoch = 0
        train_losses = []
        val_losses = []

    # Training Loop
    for epoch in range(start_epoch, int(args.epochs)):
        if distributed:
            # Ensure each replica sees a distinct subset of the dataset
            dataloaders['train'].sampler.set_epoch(epoch)

        # Training
        generator.train()
        train_loss = 0.0

        for i, (input_tensor, target_tensor) in enumerate(dataloaders['train']):
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)
            optimizer.zero_grad()
            gen_images = generator(input_tensor).squeeze(1)
            loss = loss_func(gen_images, target_tensor)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()


        lr_scheduler.step()
        train_losses.append(train_loss / len(dataloaders['train']))

        # Validation
        generator.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, (input_tensor, target_tensor) in enumerate(dataloaders['val']):
                input_tensor = input_tensor.to(device)
                target_tensor = target_tensor.to(device)

                gen_images = generator(input_tensor).squeeze(1)
                loss = loss_func(gen_images, target_tensor)
                val_loss += loss.item()


        val_losses.append(val_loss / len(dataloaders['val']))

        # Print results every N steps (eg. 10)
        if epoch % 1 == 0:
            print(
                f"Rank: {rank:2d}, "
                f"Epoch [{epoch + 1}/{args.epochs}], "
                f"Train Loss: {train_losses[-1]:.4f}, "
                f"Val Loss: {val_losses[-1]:.4f}, "
            )

    # Save checkpoint only on rank 0
    if rank == 0:
        save_checkpoint(generator,
                        optimizer,
                        lr_scheduler,
                        args.epochs,
                        train_losses,
                        val_losses,
                        os.path.join(args.checkpoint_dir, f'{args.checkpoint_id}_final_model.pth'))

    if distributed:
        dist.destroy_process_group()

    if rank == 0:
        print("Completed training!")
        print("*" * 10)


if __name__ == "__main__":
    import argparse
    import yaml
    from types import SimpleNamespace

    def convert_dict_to_namespace(d):
        for key, value in d.items():
            if isinstance(value, dict):
                d[key] = convert_dict_to_namespace(value)
        return SimpleNamespace(**d)

    parser = argparse.ArgumentParser(description='Train GeoVMS')
    parser.add_argument('--config', default='config.yaml', help='Path to YAML config file')
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    args = convert_dict_to_namespace(config)

    train(args)