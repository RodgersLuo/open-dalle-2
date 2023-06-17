# %%
import torch
import torch.nn as nn
import yaml
import sys

# %%
with open('./model_config.yml', 'r') as file:
    config = yaml.safe_load(file)
    decoder_config = config["Decoder"]
    prior_config = config["Prior"]
    clip_config = config["CLIP"]

# %%
sys.path.insert(0, 'clip')
from clip.model import CLIP

sys.path.insert(0, 'dalle2')
from prior import Prior
from decoder import UNet, Decoder
from diffusion import Diffusion
from dalle2_model import DALLE2

# %%
decoder_path = "./models copy/decoder3_cosine.pth"
prior_path = "./models/prior3_cosine.pth"
clip_path = "./models/clip3.pth"


decoder_state = torch.load(decoder_path)
prior_state = torch.load(prior_path)
clip_state = torch.load(clip_path)

decoder_config = decoder_state["config"]["Decoder"]
prior_config = prior_state["config"]["Prior"]
clip_config = clip_state["config"]["CLIP"]

# %%
# Define hyperparameters
T = decoder_config["diffusion_timesteps"]
BATCH_SIZE = decoder_config["batch_size"]
IMG_SIZE = config["img_size"]
EPOCHS = decoder_config["epochs"]
LR = decoder_config["lr"]
GRAD_CLIP = decoder_config["grad_clip"]
NULL_TEXT_EMB_RATE = decoder_config["null_text_emb_rate"]
NULL_CLIP_EMB_RATE = decoder_config["null_clip_emb_rate"]
GUIDANCE_SCALE = decoder_config["guidance_scale"]

# CLIP embedding dimension
CLIP_EMB_DIM = config["CLIP"]["embed_dim"]

# UNet
DOWN_CHANNELS = decoder_config["down_channels"]
TIME_EMB_DIM = decoder_config["time_emb_dim"]

# UNet Transformer
N_VOCAB = decoder_config["n_vocab"]
CONTEXT_LENGTH = decoder_config["context_length"]
TRANSFORMER_WIDTH = decoder_config["transformer_width"]
TRANSFORMER_LAYERS = decoder_config["transformer_layers"]
TRANSFORMER_HEADS = decoder_config["transformer_heads"]

# UNet attention block
QKV_HEADS = decoder_config["qkv_heads"]


# Create diffusion
decoder_diffusion = Diffusion(T, schedule=decoder_config["noise_schedule"])
# decoder_diffusion = Diffusion(T, schedule="linear")

# Create UNet
unet = UNet(
    down_channels=DOWN_CHANNELS,
    time_emb_dim=TIME_EMB_DIM,
    n_vocab=N_VOCAB,
    context_length=CONTEXT_LENGTH,
    transformer_width=TRANSFORMER_WIDTH,
    transformer_layers=TRANSFORMER_LAYERS,
    transformer_heads=TRANSFORMER_HEADS,
    qkv_heads=QKV_HEADS,
    clip_emb_dim=CLIP_EMB_DIM
)
# state = torch.load(decoder_config["model_path"])

# Create decoder
decoder = Decoder(unet, diffusion=decoder_diffusion, num_timesteps=T)
decoder.load_state_dict(decoder_state["model"])

# %%
T = prior_config["diffusion_timesteps"]
BATCH_SIZE = prior_config["batch_size"]
IMG_SIZE = config["img_size"]
EPOCHS = prior_config["epochs"]
LR = prior_config["lr"]

prior_diffusion = Diffusion(T, schedule=prior_config["noise_schedule"])

prior = Prior(
    clip_emb_dim=clip_config["embed_dim"],
    T=T,
    diffusion=prior_diffusion,
    clip_context_len=clip_config["context_length"],
    clip_token_dim=clip_config["transformer_width"],
    xf_layers=prior_config["xf_layers"],
    xf_heads=prior_config["xf_heads"],
)
prior.load_state_dict(prior_state["model"])


# %%
clip = CLIP(
    embed_dim=clip_config["embed_dim"],
    image_resolution=IMG_SIZE,
    vision_layers=clip_config["vision_layers"],
    vision_width=clip_config["vision_width"],
    vision_patch_size=clip_config["vision_patch_size"],
    context_length=clip_config["context_length"],
    vocab_size=clip_config["vocab_size"],
    transformer_width=clip_config["transformer_width"],
    transformer_heads=clip_config["transformer_heads"],
    transformer_layers=clip_config["transformer_layers"]
)
# state = torch.load(clip_config["model_path"])

clip.load_state_dict(clip_state["model"])

# %%
dalle2 = DALLE2(clip, prior, decoder)
dalle2.val_mode()


# %%
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

IMG_DIM = (3, IMG_SIZE, IMG_SIZE)
reverse_transforms = transforms.Compose([
    transforms.Lambda(lambda t: (t + 1) / 2),
    transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
    transforms.Lambda(lambda t: t * 255.),
    transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
    transforms.ToPILImage(),
])
def show_tensor_image(image):
    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

dalle2 = dalle2.to(device=device)

# %%

# %%
sys.path.insert(0, 'dataset')
from dataset import load_data
from torch.utils.data import DataLoader

train_data, test_data = load_data(root_dir="./data3", img_size=IMG_SIZE)
dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# %%
import pandas as pd

# %%
sys.path.insert(0, 'nn_components')
from tokenizer import tokenize

@torch.no_grad()
def sample_plot_image(decoder: Decoder, tokens, clip_emb, diffusion: Diffusion, guidance_scale=GUIDANCE_SCALE, **kwargs):
    # model.eval()
    assert tokens.shape == (1, CONTEXT_LENGTH)
    # Sample noise
    img = torch.randn((1, 3, IMG_SIZE, IMG_SIZE), device=device)
    fig = plt.figure(figsize=(15,6))
    plt.axis('off')

    num_images = 10
    stepsize = int(T/num_images)

    title = kwargs["caption"]
    assert clip_emb.shape == (1, CLIP_EMB_DIM)

    plt.title(title)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = decoder.sample_timestep(img, t, tokens, clip_emb, cf_guidance_scale=guidance_scale)
        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            fig.add_subplot(1, num_images, int(i/stepsize)+1)
            show_tensor_image(img.detach().cpu())
    plt.show()

def generate_images(n, cf_guidance_scale, all_captions):
    images = []
    captions = []
    for i in range(n):
        caption = np.random.choice(all_captions)
        captions.append(caption)
        image = dalle2((3, IMG_SIZE, IMG_SIZE), caption, cf_guidance_scale=cf_guidance_scale)
        images.append(image.detach().cpu())
        if i % 10 == 0:
            print(i)

import os

# GUIDANCE_SCALE = 3
# dir = f"evaluations/v3/cf{GUIDANCE_SCALE}/"
# # dir = "evaluations/v3/cf2/normal_cosine/"
# print(GUIDANCE_SCALE)
# %%
# images = []
# captions = []
# all_captions = pd.unique(train_data.captions)
# for i in range(3000, 15000):
#     caption = np.random.choice(all_captions)
#     captions.append(caption)
#     image = dalle2((3, IMG_SIZE, IMG_SIZE), caption, prior_diffusion, decoder_diffusion, cf_guidance_scale=GUIDANCE_SCALE)
#     image = image.detach().cpu()
#     if i % 10 == 0:
#         print(i)

#     filename = f"{i}.png"
#     image = reverse_transforms(image.squeeze(0))
#     image.save(os.path.join(dir, "images", filename))

# cs = pd.DataFrame({"caption": captions})
# cs.to_csv(os.path.join(dir, "captions.csv"))


import time

images = []
captions = pd.read_csv("evaluations/v3/reference/0shot/data.csv")["caption"].tolist()
times = []
for i in range(len(captions)):
    caption = captions[i]
    start_time = time.time()
    image = dalle2((3, IMG_SIZE, IMG_SIZE), caption, cf_guidance_scale=2)
    times.append(time.time() - start_time)

    image = image.detach().cpu()
    if i % 10 == 0:
        print(i)

    filename = f"{i}.png"
    image = reverse_transforms(image.squeeze(0))
    image.save(os.path.join("evaluations/v3/cf2/0shot", "images", filename))

print(np.mean(times), np.std(times))
# %%
cs = pd.DataFrame({"caption": captions})
cs.to_csv(os.path.join("evaluations/v3/cf2/0shot", "captions.csv"))






