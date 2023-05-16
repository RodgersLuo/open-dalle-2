import torch
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os
import yaml

from diffusion import sample_timestep, Diffusion
from unet import UNet

import sys
sys.path.insert(0, 'dataset')
from dataset import load_data

sys.path.insert(0, 'nn_components')
from tokenizer import tokenize

sys.path.insert(0, 'clip')
from model import CLIP

with open('./model_config.yml', 'r') as file:
    config = yaml.safe_load(file)
    decoder_config = config["Decoder"]

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

# The null caption token
NULL_TEXT_TOKEN = torch.zeros((1, CONTEXT_LENGTH), dtype=torch.int)

# The null CLIP embedding
NULL_CLIP_EMB = torch.zeros((1, CLIP_EMB_DIM))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

torch.cuda.empty_cache()

timestr = datetime.datetime.now().strftime("%m-%d %H:%M:%S")

classes = ('A plane', 'A car', 'a bird', 'a cat',
           'a deer', 'a dog', 'a frog', 'a horse', 'a ship', 'a truck')

def train(unet, dataloader, diffusion, clip=None):
    unet.to(device)
    optimizer = Adam(unet.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        for step, (img, txt) in enumerate(dataloader):
            # txt = [classes[i] for i in txt]

            # model.train()
            optimizer.zero_grad()

            t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()

            # tokenize text captions
            tokens = tokenize(list(txt), context_length=CONTEXT_LENGTH)
            # randomly drop tokens
            mask = torch.rand(BATCH_SIZE) < NULL_TEXT_EMB_RATE
            tokens[mask] = NULL_TEXT_TOKEN
            tokens = tokens.to(device=device)

            # obtain CLIP emmeddings
            if clip is not None:
                clip_embedding = clip.encode_image(img)
                # clip_embedding /= clip_embedding.norm(dim=1, keepdim=True)
                mask = torch.rand(BATCH_SIZE) < NULL_CLIP_EMB_RATE
                clip_embedding[mask] = NULL_CLIP_EMB
                clip_embedding = clip_embedding.to(device=device)
            else:
                clip_embedding = NULL_CLIP_EMB

            x_noisy, noise = diffusion.forward_diffusion_sample(img, t, device)
            noise_pred = unet(x_noisy, t, tokens=tokens, clip_emb=clip_embedding)
            loss = F.l1_loss(noise, noise_pred)

            # loss = get_loss(unet, img, t, tokens, diffusion, clip_emb=clip_embedding)
            loss.backward()
            torch.nn.utils.clip_grad.clip_grad_norm_(unet.parameters(), GRAD_CLIP)
            optimizer.step()

            if step == 0:
                print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
                sample_plot_image(unet, tokens[None, 0], clip_embedding[None, 0], diffusion, f"{epoch:03}(1)", caption=txt[0])
                if epoch < 10:
                    sample_plot_image(unet, tokens[None, 1], clip_embedding[None, 1], diffusion, f"{epoch:03}(2)", caption=txt[1])


# def get_loss(unet, x_0, t, tokens, diffusion, clip_emb=None):
#     x_noisy, noise = diffusion.forward_diffusion_sample(x_0, t, device)
#     noise_pred = unet(x_noisy, t, tokens, clip_emb=clip_emb)
#     return F.l1_loss(noise, noise_pred)


@torch.no_grad()
def sample_plot_image(unet, tokens, clip_emb, diffusion, filename, **kwargs):
    # model.eval()
    assert tokens.shape == (1, CONTEXT_LENGTH)
    # Sample noise
    img = torch.randn((1, 3, IMG_SIZE, IMG_SIZE), device=device)
    fig = plt.figure(figsize=(15,6))
    plt.axis('off')

    num_images = 10
    stepsize = int(T/num_images)

    title = kwargs["caption"]
    if torch.equal(tokens.detach().cpu(), NULL_TEXT_TOKEN):
        title = f"{title}, null text token"
    else:
        title = f"{title}, with text token"

    assert clip_emb.shape == (1, CLIP_EMB_DIM)
    if torch.equal(clip_emb.detach().cpu(), NULL_CLIP_EMB):
        title = f"{title}, null CLIP embedding"
    else:
        title = f"{title}, with CLIP embedding"

    plt.title(title)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t, tokens, clip_emb, unet, diffusion, cf_guidance_scale=GUIDANCE_SCALE)
        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            fig.add_subplot(1, num_images, int(i/stepsize)+1)
            show_tensor_image(img.detach().cpu())
    if filename is not None:
        os.makedirs(f"./outputs/diffusion/{timestr}", exist_ok=True)
        plt.savefig(f"./outputs/diffusion/{timestr}/{filename}")
    plt.close()


def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))


if __name__ == "__main__":
    # Create diffusion
    diffusion = Diffusion(T)

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
    param_size = 0
    for param in unet.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in unet.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

    print("Num params: ", sum(p.numel() for p in unet.parameters()))

    # Load pretrained CLIP model
    clip_config = config["CLIP"]
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
    clip.to(device="cpu")
    clip.load_state_dict(torch.load(clip_config["model_path"]))

    # Freeze CLIP model
    clip.eval()
    for param in clip.parameters():
        param.requires_grad = False

    train_data, _ = load_data(img_size=IMG_SIZE)
    dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    train(unet, dataloader, diffusion, clip=clip)

