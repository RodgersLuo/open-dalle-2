import torch
from train import *
from model import CLIP
from tokenizer import tokenize
from PIL import Image


# model = CLIP(embed_dim=128, image_resolution=128, vision_layers=(1, 1, 1, 1),
#         vision_width=256, vision_patch_size=None, context_length=77,
#         vocab_size=49408, transformer_width=64, transformer_heads=8, transformer_layers=4)

model = torch.load("models/clip.pth")
model.eval()

loader_train, loader_val, loader_test, captions = load_data()

# print(check_accuracy(loader_test, model, captions=captions))
# for i in range(len(captions)):
#     captions[i] = "a photo of " + captions[i]
i = 11

next_batch = next(iter(loader_test))
image = next_batch[0].to(device=device, dtype=dtype)[i:i+1]
t = next_batch[1][i]
print(image.size())
print(t)
text = tokenize(captions).to(device)

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
