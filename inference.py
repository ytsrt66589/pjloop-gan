import argparse
import torch
from torchvision import utils
from stylegan2.stylegan2_network import Generator
from tqdm import tqdm
import random
import sys
from melgan.modules import Generator_melgan
import yaml
import os
import librosa
import soundfile as sf
import numpy as np
import os

def read_yaml(fp):
    with open(fp) as file:
        return yaml.load(file, Loader=yaml.Loader)
    
def generate(args, g_ema, device, mean_latent):
    epoch = args.ckpt.split('.')[0]

    os.makedirs(f'{args.store_path}', exist_ok=True)
    os.makedirs(f'{args.store_path}/mel_64_192', exist_ok=True)
    feat_dim = 64
    mean_fp = f'{args.data_path}/mean.mel.npy'
    std_fp = f'{args.data_path}/std.mel.npy'
    mean = torch.from_numpy(np.load(mean_fp)).float().view(1, feat_dim, 1).to(device)
    std = torch.from_numpy(np.load(std_fp)).float().view(1, feat_dim, 1).to(device)
    
    #################################
    ## vocoder 
    #################################
    vocoder_config_fp = os.path.join(args.vocoder_folder, 'args.yml')
    vocoder_config = read_yaml(vocoder_config_fp)

    n_mel_channels = vocoder_config.n_mel_channels
    
    ngf = vocoder_config.ngf
    n_residual_layers = vocoder_config.n_residual_layers
    sr = 16000
    vocoder = Generator_melgan(n_mel_channels, ngf, n_residual_layers).to(device)
    vocoder.eval()

    vocoder_param_fp = os.path.join(args.vocoder_folder, 'best_netG.pt')
    vocoder.load_state_dict(torch.load(vocoder_param_fp))
    
    
    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.pics)):
            sample_z = torch.randn(args.sample, args.latent, device=device)
            sample, _ = g_ema(
                [sample_z], truncation=args.truncation, truncation_latent=mean_latent
            )
            sample = sample.permute(0, 1, 3, 2)
            
            np.save(f'{args.store_path}/mel_64_192/{i}.npy', sample.squeeze().data.cpu().numpy())

            utils.save_image(
                sample,
                f"{args.store_path}/{str(i).zfill(6)}.png",
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )

            de_norm = sample.squeeze(0) * std + mean
            audio_output = vocoder(de_norm)
            p = np.full((1280, ), 1e-8)
            sf.write(f'{args.store_path}/{i}.wav', np.concatenate((audio_output.squeeze().detach().cpu().numpy(), p), axis=None), sr)
            print('generate {}th wav file'.format(i))

if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--size", type=int, default=64, help="output image size of the generator"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--pics", type=int, default=20, help="number of images to be generated"
    )
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="stylegan2-ffhq-config-f.pt",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="path store the std and mean of mel",
    )
    parser.add_argument(
        "--store_path",
        type=str,
        help="path store the generated audio",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--vocoder_folder",
        type=str,
        default='melgan/drum_vocoder',
        help="folder of the vocoder",
    )
    parser.add_argument("--style_mixing", action = "store_true")
    parser.add_argument('--n_row', type=int, default=3, help='number of rows of sample matrix')
    parser.add_argument('--n_col', type=int, default=5, help='number of columns of sample matrix')
    args = parser.parse_args()

    args.latent = 32
    args.n_mlp = 6
    
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    
    checkpoint = torch.load(args.ckpt)
    g_ema.load_state_dict(checkpoint["g_ema"])

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    # Generate audio
    generate(args, g_ema, device, mean_latent)