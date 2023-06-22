import torch
from train import *
from model import CLIP
from PIL import Image
import yaml

sys.path.insert(0, 'dataset')
sys.path.insert(0, 'nn_components')
from tokenizer import tokenize


with open('./model_config.yml', 'r') as file:
    config = yaml.safe_load(file)
    decoder_config = config["Decoder"]

clip_config = config["CLIP"]
model = CLIP(
    embed_dim=clip_config["embed_dim"],
    image_resolution=config["img_size"],
    vision_layers=clip_config["vision_layers"],
    vision_width=clip_config["vision_width"],
    vision_patch_size=clip_config["vision_patch_size"],
    context_length=clip_config["context_length"],
    vocab_size=clip_config["vocab_size"],
    transformer_width=clip_config["transformer_width"],
    transformer_heads=clip_config["transformer_heads"],
    transformer_layers=clip_config["transformer_layers"]
)
model.to(device=device)
model.load_state_dict(torch.load(clip_config["model_path"]))

# Freeze CLIP model
model.eval()
for param in model.parameters():
    param.requires_grad = False

loader_train, loader_val, loader_test, captions = load_dataset()

# print(check_accuracy(loader_test, model, captions=captions))
# for i in range(len(captions)):
#     captions[i] = "a photo of " + captions[i]
i = 18

next_batch = next(iter(loader_test))
image = next_batch[0].to(device=device, dtype=dtype)[i:i+1]
t = next_batch[1][i]
print(image.size())
print(t)
text = tokenize(captions, context_length=clip_config["context_length"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(5)

    # Print the result
    print("\nTop predictions:\n")
    for value, index in zip(values, indices):
        print(f"{captions[index]}: {100 * value.item():.2f}%")
