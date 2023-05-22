import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import yaml
import wandb

from diffusion import sample_timestep, Diffusion
from prior import DiffusionPriorNetwork

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

wandb.init(
    # set the wandb project where this run will be logged
    project="open-dalle-2-prior",
    # track hyperparameters and run metadata
    config=config,
    mode="disabled"
)

LR = 0.01
EPOCHS = 100
T = 200
BATCH_SIZE = 512
CONTEXT_LENGTH = 33
IMG_SIZE = 32


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

torch.cuda.empty_cache()

def train(prior, train_dataloader, val_dataloader, diffusion, clip=None):
    prior.to(device)
    wandb.watch(prior, log="all", log_freq=10)

    optimizer = Adam(prior.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        for step, (img, txt, image_embedding, text_embedding) in enumerate(train_dataloader):
            # txt = [classes[i] for i in txt]

            prior.train()
            optimizer.zero_grad()

            t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()

            # tokenize text captions
            tokens = tokenize(list(txt), context_length=CONTEXT_LENGTH)

            # img = img.to(device=device)
            tokens = tokens.to(device=device)
            image_embedding = image_embedding.to(device=device)
            text_embedding = text_embedding.to(device=device)

            # # obtain CLIP emmeddings
            # image_embedding = clip.encode_image(img)
            # text_embedding = clip.encode_text(tokens)
            # clip_embedding /= clip_embedding.norm(dim=1, keepdim=True)

            img_emb_noisy, noise = diffusion.forward_diffusion_sample(image_embedding, t, device)
            img_emb_pred = prior(img_emb_noisy, t, text_embed=text_embedding, text_encodings=tokens)
            loss = F.mse_loss(img_emb_pred, image_embedding)

            loss.backward()
            # torch.nn.utils.clip_grad.clip_grad_norm_(prior.parameters(), GRAD_CLIP)
            optimizer.step()

            if step == 0:
                print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
                wandb.log({"train_loss": loss.item()})

                if epoch % 10 == 0 or epoch == EPOCHS - 1:
                    validate(prior, val_dataloader, diffusion, full_sample=True)
                else:
                    validate(prior, val_dataloader, diffusion, full_sample=False)

        if epoch == EPOCHS - 1:
            a = 1

def validate(prior, val_dataloader, diffusion, full_sample=False):
    prior.eval()  # set model to evaluation mode
    with torch.no_grad():
        img, txt, image_embedding, text_embedding = next(iter(val_dataloader))
        # tokenize text captions
        tokens = tokenize(list(txt), context_length=CONTEXT_LENGTH)

        tokens = tokens.to(device=device)
        image_embedding = image_embedding.to(device=device)
        text_embedding = text_embedding.to(device=device)

        if full_sample:
            img_emb_pred = prior.sample(diffusion, T, text_embedding, tokens)
            loss = F.mse_loss(img_emb_pred, image_embedding).item()
            print(f"Full sample validation Loss: {loss}")
            wandb.log({"full_sampmle_val_loss": loss})
        else:
            t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()

            img_emb_noisy, noise = diffusion.forward_diffusion_sample(image_embedding, t, device)
            img_emb_pred = prior(img_emb_noisy, t, text_embed=text_embedding, text_encodings=tokens)
            loss = F.mse_loss(img_emb_pred, image_embedding).item()
            print(f"Validation Loss: {loss}")
            wandb.log({"val_loss": loss})



if __name__ == "__main__":
    # Create diffusion
    diffusion = Diffusion(T)

    # Create Prior
    prior = DiffusionPriorNetwork(
        dim=config["CLIP"]["embed_dim"],
        num_timesteps=T,
        max_text_len=CONTEXT_LENGTH,
        depth=4,
        dim_head=32,
        heads=8,
        ff_mult=4
    )
    param_size = 0
    for param in prior.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in prior.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

    print("Num params: ", sum(p.numel() for p in prior.parameters()))

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

    train_data, val_data = load_data(img_size=IMG_SIZE, clip=clip, context_length=CONTEXT_LENGTH, normalize_clip_embeddings=True)
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    train(prior, train_dataloader, val_dataloader, diffusion, clip=clip)
