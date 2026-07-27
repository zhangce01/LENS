import os, time, base64, requests, json, sys, datetime, argparse
from itertools import product

from PIL import Image
import cv2

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T

from API_CLIP.clip_prs.utils.factory import create_model_and_transforms, get_tokenizer
from API_CLIP.hook import hook_prs_logger
try:
    from DatasetManager.dataloader import get_data, Disposable_Dataloader
except ImportError:
    # DatasetManager is optional, only needed for batch processing
    pass

def toImg(t):
    return T.ToPILImage()(t)

def invtrans(mask, image, method = Image.BICUBIC):
    return mask.resize(image.size, method)

def merge(mask, image, grap_scale = 200):
    gray = np.ones((image.size[1], image.size[0], 3))*grap_scale
    image_np = np.array(image).astype(np.float32)[..., :3]
    mask_np = np.array(mask).astype(np.float32)
    mask_np = mask_np / 255.0
    blended_np = image_np * mask_np[:, :, None]  + (1 - mask_np[:, :, None]) * gray
    blended_image = Image.fromarray((blended_np).astype(np.uint8))
    return blended_image

def normalize(mat, method = "max"):
    if method == "max":
        return (mat.max() - mat) / (mat.max() - mat.min())
    elif method == "min":
        return (mat - mat.min()) / (mat.max() - mat.min())
    else:
        raise NotImplementedError

def enhance(mat, coe=10):
    mat = mat - mat.mean()
    mat = mat / mat.std()
    mat = mat * coe
    mat = torch.sigmoid(mat)
    mat = mat.clamp(0,1)
    return mat

def get_model(model_name = "ViT-L-14-336", layer_index = 23, device = 'cuda:0'): # "ViT-L-14", "ViT-B-32"
    ## Hyperparameters
    pretrained = 'openai' # 'laion2b_s32b_b79k'

    ## Loading Model
    model, _, preprocess = create_model_and_transforms(model_name, pretrained=pretrained)
    model.to(device)
    model.eval()
    context_length = model.context_length
    vocab_size = model.vocab_size
    tokenizer = get_tokenizer(model_name)

    # print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    # print("Context length:", context_length)
    # print("Vocab size:", vocab_size)
    # print("Len of res:", len(model.visual.transformer.resblocks))

    prs = hook_prs_logger(model, device, layer_index)

    return model, prs, preprocess, device, tokenizer

def gen_mask(model, prs, preprocess, device, tokenizer, image_path_or_pil_images, questions):
    ## Load image
    images = []
    image_pils = []
    for image_path_or_pil_image in image_path_or_pil_images:
        if isinstance(image_path_or_pil_image, str):
            image_pil = Image.open(image_path_or_pil_image)
        elif isinstance(image_path_or_pil_image, Image.Image):
            image_pil = image_path_or_pil_image
        else:
            raise NotImplementedError
        image = preprocess(image_pil)[np.newaxis, :, :, :]
        images.append(image)
        image_pils.append(image_pil)
    image = torch.cat(images, dim = 0).to(device)

    ## Run the image:
    prs.reinit()
    with torch.no_grad():
        representation = model.encode_image(image, 
                                            attn_method='head', 
                                            normalize=False)  
                         
        attentions, mlps = prs.finalize(representation)  

    ## Get the texts
    lines = questions if isinstance(questions, list) else [questions]
    # print(lines[0])
    texts = tokenizer(lines).to(device)  # tokenize
    class_embeddings = model.encode_text(texts)
    class_embedding = F.normalize(class_embeddings, dim=-1)

    attention_map = attentions[:, 0, 1:, :]
    attention_map = torch.einsum('bnd,bd->bn', attention_map, class_embedding)
    HW = int(np.sqrt(attention_map.shape[1]))
    batch_size = attention_map.shape[0]
    attention_map = attention_map.view(batch_size,HW,HW)
    # print(HW)

    token_map = torch.einsum('bnd,bd->bn', mlps[:,0,:,:], class_embedding)
    token_map = token_map.view(batch_size,HW,HW)

    return image_pils, attention_map, token_map

def merge_mask(cls_mask, patch_mask, kernel_size = 3, enhance_coe = 10):
    cls_mask = normalize(cls_mask, "min")
    cls_mask = enhance(cls_mask, coe = enhance_coe)

    patch_mask = normalize(patch_mask, "max")

    assert kernel_size % 2 == 1
    padding_size = int((kernel_size - 1) / 2)
    conv = torch.nn.Conv2d(1,1,kernel_size = kernel_size, padding = padding_size, padding_mode = "replicate", stride = 1, bias = False)
    conv.weight.data = torch.ones_like(conv.weight.data) / kernel_size**2
    conv.to(cls_mask.device)

    cls_mask = conv(cls_mask.unsqueeze(0))[0]

    patch_mask = conv(patch_mask.unsqueeze(0))[0]

    mask = normalize(cls_mask + patch_mask - cls_mask * patch_mask, "min")

    return mask

def blend_mask(image, cls_mask, patch_mask, key, enhance_coe, kernel_size, interpolate_method, grayscale, folder):
    mask = merge_mask(cls_mask, patch_mask, kernel_size = kernel_size, enhance_coe = enhance_coe)
    mask = toImg(mask.detach().cpu().unsqueeze(0))
    mask = invtrans(mask, image, method = interpolate_method)
    merged_image = merge(mask.convert("L"), image.convert("RGB"), grayscale).convert("RGB")

    file_name = os.path.join(folder, f"{key}.jpg")
    merged_image.save(file_name)
    print(file_name)
    return merged_image

def api_clip(frame, prompt, model=None, prs=None, preprocess=None, device=None, tokenizer=None, 
             enhance_coe=10, kernel_size=3, interpolate_method_name="BICUBIC", grayscale=200, 
             save_folder=None, frame_key=None):
    """
    Apply CLIP-based masking to a frame based on a text prompt.
    
    Args:
        frame: Input frame - can be PIL Image, numpy array (H,W,C) or (C,H,W), or torch.Tensor (C,H,W)
        prompt: Text prompt for masking
        model: CLIP model (if None, will use global model)
        prs: PRS logger (if None, will use global prs)
        preprocess: Preprocessing function (if None, will use global preprocess)
        device: Device to run on (if None, will use global device)
        tokenizer: Tokenizer (if None, will use global tokenizer)
        enhance_coe: Enhancement coefficient for mask
        kernel_size: Kernel size for mask smoothing
        interpolate_method_name: Interpolation method name (e.g., "BICUBIC")
        grayscale: Grayscale value for masked regions
        save_folder: Folder to save masked frames (optional)
        frame_key: Key for saving file (optional)
    
        Returns:
            numpy array: Masked frame as numpy array (H, W, C)
    """
    # Convert frame to PIL Image
    if isinstance(frame, torch.Tensor):
        frame_np = frame.cpu().numpy()
    elif isinstance(frame, np.ndarray):
        frame_np = frame
    else:
        frame_np = np.array(frame)
    
    # Handle different tensor shapes: (C, H, W) or (H, W, C) or (H, W)
    if len(frame_np.shape) == 3:
        if frame_np.shape[0] == 3 or frame_np.shape[0] == 1:
            # (C, H, W) format, convert to (H, W, C)
            frame_np = np.transpose(frame_np, (1, 2, 0))
        elif frame_np.shape[2] == 3 or frame_np.shape[2] == 1:
            # Already (H, W, C) format
            pass
        else:
            raise ValueError(f"Unexpected frame shape: {frame_np.shape}")
    elif len(frame_np.shape) == 2:
        # (H, W) grayscale, add channel dimension
        frame_np = frame_np[:, :, np.newaxis]
    else:
        raise ValueError(f"Unexpected frame shape: {frame_np.shape}")
    
    # Ensure RGB format (3 channels)
    if frame_np.shape[2] == 1:
        # Grayscale to RGB
        frame_np = np.repeat(frame_np, 3, axis=2)
    elif frame_np.shape[2] == 4:
        # RGBA to RGB
        frame_np = frame_np[:, :, :3]
    
    # Normalize to uint8 if needed
    if frame_np.dtype != np.uint8:
        if frame_np.max() <= 1.0:
            frame_np = (frame_np * 255).astype(np.uint8)
        else:
            frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
    
    # Convert to PIL Image
    frame_pil = Image.fromarray(frame_np, mode='RGB')
    
    # Generate mask
    image_pils, attention_map, token_map = gen_mask(
        model, prs, preprocess, device, tokenizer, 
        [frame_pil], [prompt]
    )
    
    # Extract masks for the first (and only) image in the batch
    cls_mask = attention_map[0]  # Shape: (HW, HW)
    patch_mask = token_map[0]     # Shape: (HW, HW)
    
    # Convert interpolate_method_name string to Image method
    interpolate_method = getattr(Image, interpolate_method_name)
    
    # Blend mask with image
    if save_folder is not None and frame_key is not None:
        os.makedirs(save_folder, exist_ok=True)
        masked_frame = blend_mask(
            frame_pil, 
            cls_mask, 
            patch_mask, 
            key=frame_key,
            enhance_coe=enhance_coe, 
            kernel_size=kernel_size, 
            interpolate_method=interpolate_method, 
            grayscale=grayscale,
            folder=save_folder
        )
    else:
        # Use a simplified blend_mask that doesn't save
        mask = merge_mask(cls_mask, patch_mask, kernel_size=kernel_size, enhance_coe=enhance_coe)
        mask = toImg(mask.detach().cpu().unsqueeze(0))
        mask = invtrans(mask, frame_pil, method=interpolate_method)
        masked_frame = merge(mask.convert("L"), frame_pil.convert("RGB"), grayscale).convert("RGB")
    
    masked_frame_np = np.array(masked_frame)
    masked_frame_np = masked_frame_np.transpose(2, 0, 1)
    # to torch tensor
    masked_frame = torch.from_numpy(masked_frame_np)
    
    return masked_frame

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mmvet")
    parser.add_argument("--model_name", type=str, default="ViT-L-14-336")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--layer_index", type=int, default=22)
    parser.add_argument("--range", type=int, nargs = 2, default=(0,0))
    parser.add_argument("--output_folder", type=str, default = "../../experiments")
    parser.add_argument("--interpolate_method_name", type=str, default = "LANCZOS")
    parser.add_argument("--enhance_coe", type=int, default = 5)
    parser.add_argument("--kernel_size", type=int, default = 3)
    parser.add_argument("--grayscale", type=int, default = 0)
    args = parser.parse_args()

    exp_folder = f"{args.output_folder}/APICLIP_{args.dataset}_{args.model_name}_{args.layer_index}"

    model, prs, preprocess, device, tokenizer = get_model(model_name = args.model_name, layer_index = args.layer_index)

    dataset = get_data(args.dataset)
    batch_size = args.batch_size
    dataset_loader = Disposable_Dataloader(dataset, batch_size, args.range)

    enhance_coe = args.enhance_coe
    kernel_size = args.kernel_size
    grayscale = args.grayscale
    interpolate_method_name = args.interpolate_method_name
    interpolate_method = getattr(Image, interpolate_method_name)
    mask_image_folder = os.path.join(exp_folder, f"{enhance_coe}_{kernel_size}_{interpolate_method_name}_{grayscale}")
    os.makedirs(mask_image_folder, exist_ok = True)

    for keys, image_path_or_pil_images, questions in dataset_loader:

        with torch.no_grad():
            image_pils, cls_masks, patch_masks = gen_mask(model, prs, preprocess, device, tokenizer, image_path_or_pil_images, questions)

        for key, image, cls_mask, patch_mask in zip(keys, image_pils, cls_masks, patch_masks):
            merged_image = blend_mask(image, cls_mask, patch_mask, key, enhance_coe, kernel_size, interpolate_method, grayscale, mask_image_folder)