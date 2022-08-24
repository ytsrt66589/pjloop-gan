import argparse
import os
import numpy as np
from numpy import real
import torch
import utils as function
import loss as loss
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms, utils
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from stylegan2.stylegan2_network import Generator
from pg_modules.discriminator import  SingleProjectedDiscriminator, FusionProjectedDiscriminator
from datasets.dataset import MultiResolutionDataset

def train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device):
    os.makedirs(args.sample_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    loader = function.sample_data(loader)
    pbar = range(args.iter)

    pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)
    
    mean_path_length = 0
    
    d_loss_val = 0
    g_loss_val = 0
    loss_dict = {}

    g_module = generator
    d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))
    
    sample_z = torch.randn(args.n_sample, args.latent, device=device)
    for idx in pbar:
        i = idx + args.start_iter
        
        if i > args.iter : 
            print('Done!')
            break
        
        real_img= next(loader)
        real_img = real_img.to(device)

        mean_fp = os.path.join(args.path, f'mean.mel.npy')
        std_fp = os.path.join(args.path, f'std.mel.npy')
        feat_dim = 64
        mean = torch.from_numpy(np.load(mean_fp)).float().cuda().view(1, feat_dim, 1)
        std = torch.from_numpy(np.load(std_fp)).float().cuda().view(1, feat_dim, 1)
        real_img = (real_img - mean) / std

        if i == 0 :
            utils.save_image(
                real_img,
                f"{args.sample_dir}/real.png",
                nrow=int(args.n_sample ** 0.5),
                normalize=True,
                range=(-1, 1),
            )
        real_img = real_img.permute(0, 1, 3, 2)        
        real_img = real_img[:, :, 0:192, :]
        
        
        # G : False, D : True 
        function.requires_grad(generator, False)
        function.requires_grad(discriminator, True)

        if args.fusion:
            discriminator.feature_network_domain.requires_grad_(False)
            discriminator.feature_network_general.requires_grad_(False)
        else:
            discriminator.feature_network.requires_grad_(False)
        
        
        noise = function.mixing_noise(args.batch, args.latent, args.mixing, device)
        
        fake_img, _ = generator(noise)
        
        real_img_aug = real_img
        fake_pred = discriminator(fake_img)
        tmp = real_img_aug.detach().requires_grad_(False)
        real_pred = discriminator(tmp)
        d_loss = loss.d_loss(real_pred, fake_pred)
        
        dloss = d_loss 
        loss_dict["d"] = dloss
        discriminator.zero_grad()
        dloss.backward()
        d_optim.step()
        

        # G: T, D : F
        function.requires_grad(generator, True)
        function.requires_grad(discriminator, False)
        if args.fusion:
            discriminator.feature_network_domain.requires_grad_(False)
            discriminator.feature_network_general.requires_grad_(False)
        else:
            discriminator.feature_network.requires_grad_(False)

        noise = function.mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, _ = generator(noise)
        
        fake_pred = discriminator(fake_img)
        g_loss = loss.g_loss(fake_pred)    
        loss_dict["g"] = g_loss
        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        function.accumulate(g_ema, g_module, accum)        
        d_loss_val = loss_dict["d"].mean().item()
        g_loss_val = loss_dict["g"].mean().item()
        
        
        pbar.set_description(
            (
                f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f};"
            )
        )

        if wandb and args.wandb:
            wandb.log(
                {
                    "Generator": g_loss_val,
                    "Discriminator": d_loss_val,
                }
            )

        if i % 100 == 0:
            with torch.no_grad():
                g_ema.eval()
                sample, _ = g_ema([sample_z])
                utils.save_image(
                    sample.permute(0, 1, 3, 2),
                    f"{args.sample_dir}/{str(i).zfill(6)}.png",
                    nrow=int(args.n_sample ** 0.5),
                    normalize=True,
                    range=(-1, 1),
                )
                
        if i % 500 == 0:
            torch.save(
                {
                    "g": g_module.state_dict(),
                    "d": d_module.state_dict(),
                    "g_ema": g_ema.state_dict(),
                    "g_optim": g_optim.state_dict(),
                    "d_optim": d_optim.state_dict(),
                    "args": args,
                },
                f"{args.checkpoint_dir}/{str(i).zfill(6)}.pt",
            )

if __name__ == "__main__":
    device = "cuda"
    print('start')

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    parser.add_argument("path", type=str, help="path to the lmdb dataset")
    parser.add_argument(
        "--iter", type=int, default=800000, help="total training iterations"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="batch sizes for each gpus"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=64,
        help="number of the samples generated during training",
    )
    parser.add_argument(
        "--size", type=int, default=256, help="image sizes for the model"
    )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=16,
        help="interval of the applying r1 regularization",
    )
    parser.add_argument(
        "--g_reg_every",
        type=int,
        default=4,
        help="interval of the applying path length regularization",
    )
    parser.add_argument(
        "--mixing", type=float, default=0.9, help="probability of latent code mixing"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="use weights and biases logging"
    )
    parser.add_argument(
        "--sample_dir",
        type=str,
        default='sample',
        help="sample directory",
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default='checkpoint',
        help="checkpoint directory",
    )

    parser.add_argument(
        "--loops_type",
        type=str,
        default="drums",
        help="[drums, synth]",
    )

    parser.add_argument(
        "--fusion",
        type=str,
        default=None,
        help="[fusion] method or not",
    )

    parser.add_argument(
        "--pre_trained_model_type",
        type=str,
        default="vgg",
        help="[vgg, autotagging, loops_genre]",
    )

    parser.add_argument(
        "--pre_trained_model_path",
        type=str,
        default=None,
        help="[pre-trained model path]",
    )

    args = parser.parse_args()

    args.latent = 32
    args.n_mlp = 6
    args.start_iter = 0

    # Model 
    # generator 
    generator = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    # discriminator 
    if args.fusion:
        print('FUSION')
        discriminator = FusionProjectedDiscriminator(
            "vgg",
            args.pre_trained_model_type,
            args.pre_trained_model_path,
            args.loops_type
        ).cuda()
    else:
        print('SINGLE')
        discriminator = SingleProjectedDiscriminator(
            args.pre_trained_model_type,
            args.pre_trained_model_path,
            args.loops_type
        ).cuda()

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema.eval()
    
    function.accumulate(g_ema, generator, 0)
    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)
    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )
    
    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        
        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])

        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    
    dataset = MultiResolutionDataset(args.path, transform)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=function.data_sampler(dataset, shuffle=True, distributed=False),
        num_workers=16,
        drop_last=True,
    )

    if wandb is not None and args.wandb:
        wandb.init(project="stylegan 2 w projected gan")

    train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device)
