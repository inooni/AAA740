import torch
import numpy as np
import os
import random

from PIL import Image, ImageDraw
from scripts.sample_diffusion import load_model
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from einops import rearrange
from ldm.util import instantiate_from_config, log_txt_as_img
from tqdm import tqdm, trange
from ldm.data.coco import CocoValidation
from ldm.data.laion import LaionValidation
from ldm.models.diffusion.ddim import DDIMSampler

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def ldm_cond_sample(config_path, ckpt_path, dataset, batch_size, model_name, caption_style):
    
    config = OmegaConf.load(config_path)
    outpath = '/home/inho/latent-diffusion/results/' + model_name

    #fixed seed
    seed = 42
    deterministic = True

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model = load_model_from_config(config, ckpt_path)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    num_samples = 5000 #coco

    os.makedirs(outpath, exist_ok=True)

    sample_path = os.path.join(outpath, "samples")
    caption_path = os.path.join(outpath, "caption")
    os.makedirs(sample_path, exist_ok=True)
    os.makedirs(caption_path, exist_ok=True)

    ddim_eta = 0.0
    n_iter = 1
    scale = 5.0
    ddim_steps = 50
    H=256
    W=256
    
    sampler = DDIMSampler(model)

    for i, x in enumerate(iter(dataloader)):

        prompt = x[caption_style]

        with torch.no_grad():
            with model.ema_scope():
                uc = None
                if scale != 1.0:
                    uc = model.get_learned_conditioning(batch_size * [""])
                for n in trange(n_iter, desc="Sampling"):
                    c = model.get_learned_conditioning(prompt) # change
                    shape = [4, H//8, W//8]
                    samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                    conditioning=c,
                                                    batch_size=batch_size,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=uc,
                                                    eta=ddim_eta)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

                    for k, x_sample in enumerate(x_samples_ddim):
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, x['image_id'][k]+".png"))

                        prompt_image = log_txt_as_img((256, 256),[prompt[k]]).squeeze(0)
                        prompt_image = 255 * (rearrange(prompt_image.cpu().numpy(), 'c h w -> h w c') + 1)/2
                        Image.fromarray(prompt_image.astype(np.uint8)).save(os.path.join(caption_path, x['image_id'][k]+".png"))


        if (i+1)*batch_size >= num_samples:
            break


if __name__ == '__main__':

    i = 7

    config_path = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
    ckpt_path = ["/home/inho/latent-diffusion/models/ldm/text2img-large/model.ckpt",
    "/home/inho/latent-diffusion/logs/2023-12-06T00-28-27_simple_prompt_image_space/checkpoints/last.ckpt",
    "/home/inho/latent-diffusion/logs/2023-12-06T00-26-41_simple_prompt_train_latent_space/checkpoints/last.ckpt",
    "/home/inho/latent-diffusion/logs/2023-12-06T22-30-16_simple_prompt_test_latent_space_random/checkpoints/epoch=000217.ckpt"
    ]
    model_name = ['baseline_scale5','baseline_prompt_scale5','img_space_scale5','img_space_prompt_scale5','latent_space_train','latent_space_train_prompt','latent_space_test_random_scale5','latent_space_test_random_prompt_scale5']
    caption_style = ['caption_ori','caption_prompt']

    dataset = CocoValidation(size=256)

    ldm_cond_sample(config_path, ckpt_path[i//2], dataset, 20, model_name[i], caption_style[i%2])
