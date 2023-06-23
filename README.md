# Open DALL-E 2: a simplified implementation of DALL-E 2.

This is a implementation of OpenAI's DALL-E 2 \[[Link](https://openai.com/dall-e-2)\] [[Paper](https://arxiv.org/abs/2204.06125)] in PyTorch. This implementation is suitable for simple text-to-image generation tasks. 



**Generated samples on CIFAR-10 dataset:**
<img width="1613" alt="image" src="https://github.com/RodgersLuo/open-dalle-2/assets/23311201/aa684850-42fd-4bea-8764-c5d2318ceb21">


**Generated samples on custom geometric shapes dataset:**

<img width="1257" alt="image" src="https://github.com/RodgersLuo/open-dalle-2/assets/23311201/d608873e-36cc-4342-93cb-c5e21fe4c03b">

## Training
The full pipeline consists of 3 models: CLIP [[Paper](https://arxiv.org/abs/2103.00020)], DALL-E 2 prior and DALL-E 2 decoder.

### CLIP
CLIP is a zero-shot model that learns a shared, multimodal latent representation of text captions and images. Unlike standard image classification models that use a feature extraction network and a final linear classification network, CLIP uses an image encoder and a text encoder to obtain pairs of shared embeddings of images and texts in the latent space. 

To train DALL-E 2, you need train CLIP first. To train CLIP, run

```
python clip/train.py
```

You have to specify the dataset path and the path where the final model is saved in <code>model_config.yml</code>.

### Prior
The prior generates the CLIP image embedding based on the text caption.

To train the prior, run
```
python dalle2/train_prior.py
```

similar to CLIP, you have to specify the dataset path and model saving path in <code>model_config.yml</code>.

### Decoder
The DALL-E 2 decoder is used to generate images conditioned on CLIP image embeddings and text captions. 

To train the decoder, run
```
python dalle2/train_decoder.py
```

Do not forget to specify the paths in <code>model_config.yml</code>.


## Sampling
The example below shows how to sample images from texts

```python
# Initialise and load CLIP
clip = CLIP(...)
clip_path = ...
clip.load_state_dict(clip_path)

# Initialise and load prior
prior = Prior(...)
prior_path = ...
prior.load_state_dict(prior_path)

# Initialise and load decoder
decoder = Decoder(...)
decoder_path = ...
decoder.load_state_dict(decoder_path)

# Initialise DALL-E 2
dalle2 = DALLE2(clip, prior, decoder)

# Set DALL-E 2 to evaluation mode
dalle2.val_mode()

# Sample the image from text caption, cf_guidance_scale is the classifier-free guidance scale
image_size = (3, 32, 32)
image = dalle2(image_size, text="a small black square and a large gold pentagon", cf_guidance_scale=2)
```
