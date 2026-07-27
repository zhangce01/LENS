import math
import torch
import numpy as np
import os
import json
from transformers.trainer_pt_utils import IterableDatasetShard
import datetime
from tqdm import tqdm
import argparse
from itertools import chain
from utils.data import (
    EvalDatasetMLVU,
    EvalDatasetVideoMME,
    EvalDatasetLongVideoBench,
    EvalDatasetEgoschemaSubset,
    EvalDatasetNExTQA,
    get_subtitles,
)
from utils.lens import LENS
from utils.sampling import clip_based_sampling, uniform_sampling, graph_based_sampling, blip2_based_sampling, blip_based_sampling
import cv2
from torch import distributed as dist
from utils.config import get_args
import sys
from utils.prompts import *
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, Blip2ForImageTextRetrieval, Blip2Processor, BlipProcessor, BlipForImageTextRetrieval
from PIL import Image
import sys

args = get_args()
dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
torch.distributed.barrier()
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
world_size = torch.distributed.get_world_size()
world_rank = torch.distributed.get_rank()
checkpoint_dir = os.path.join(f"{args.output_path}/{args.model_name}/{args.task}")
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_file = os.path.join(checkpoint_dir, f"cuda:{world_rank}.json")

lens = LENS(args)
processed_identifiers = set() 
output = []
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'r') as f:
        checkpoint_data = json.load(f)
        output = checkpoint_data.get('output', [])
        for item in output:
            if "video_name" in item and "question" in item:
                processed_identifiers.add((item["video_name"], item["question"]))
    print(f"Rank {world_rank}: Resuming with {len(output)} already processed question-video pairs.")

if args.task == "mlvu":
    dataset = EvalDatasetMLVU(data_path=args.data_path)
elif args.task == "videomme":
    dataset = EvalDatasetVideoMME(data_path=args.data_path)
elif args.task == "lvb":
    dataset = EvalDatasetLongVideoBench(data_path=args.data_path)
elif args.task == "egoschema":
    dataset = EvalDatasetEgoschemaSubset(data_path=args.data_path)
elif args.task == "nextqa":
    dataset = EvalDatasetNExTQA(data_path=args.data_path)

shard_dataset = IterableDatasetShard(
    dataset,
    batch_size=1,
    num_processes=world_size,
    process_index=world_rank,
)

torch.distributed.barrier()
total_videos_for_rank = len(list(shard_dataset))
pbar = tqdm(shard_dataset, total=total_videos_for_rank, desc=f"Rank {world_rank} Processing Videos")

# if args.itm_model_name == "CLIP":
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
if args.itm_model_name == "BLIP2":  # ITM, blip2-itm-vit-g
    blip2_model = Blip2ForImageTextRetrieval.from_pretrained("Salesforce/blip2-itm-vit-g").to(device)
    blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-itm-vit-g")
if args.itm_model_name == "BLIP":
    blip_model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-large-coco").to(device)
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-large-coco")
    
for line in pbar:
    video_name = line.get("video_name", None)
    answer = line.get("answer", None)
    prompt = line.get("prompt", None)
    question = line.get("question", None)
    task_type = line.get("task_type", None)
    video_path = line.get("video_path", None)
    candidates = line.get("candidates", None)
    subtitle_path = line.get("subtitle", None)
    duration = line.get("duration", None)
    
    

    current_identifier = (video_name, question)
    if current_identifier in processed_identifiers:
        continue

    if not os.path.exists(video_path):
        print(video_path)
        continue
    # try:
    raw_video, _, _, frame_idx, fps, video_inputs, size_list = lens.load_video(video_path, args)
    if type(raw_video) is list:
        raw_video = raw_video[0]

    subtitles = get_subtitles(subtitle_path, len(video_inputs[0]), fps=args.fps, data=line)
    
    # if duration == "long":
    #     break
    # T = len(raw_video)
    # print(f"Video {video_name} has {T} frames at {fps} fps.", flush=True)

    # UNIFORM
    # pre_selected_indices = uniform_sampling(raw_video, question, 128)
    # sampled_video = raw_video[pre_selected_indices]
    # sampled_video = raw_video
    # T = len(sampled_video)

    # CLIP-BASED SELECTION
    if args.itm_model_name == "CLIP":
        selected_indices_1, scores = clip_based_sampling(clip_model, clip_processor, raw_video, prompt, args.num_frames)
    elif args.itm_model_name == "BLIP2":
        selected_indices_1, scores = blip2_based_sampling(blip2_model, blip2_processor, raw_video, question, args.num_frames)
    elif args.itm_model_name == "BLIP":
        selected_indices_1, scores = blip_based_sampling(blip_model, blip_processor, raw_video, question, args.num_frames)
    else:
        raise ValueError(f"Invalid ITM model name: {args.itm_model_name}")
    
    
    scores = scores[selected_indices_1]
    # rank selected indices
    level_1_budget = args.num_frames // 2
    level_2_budget = args.num_frames // 4
    level_3_budget = args.num_frames - level_1_budget - level_2_budget
    
    sorted_indices = torch.argsort(scores, descending=True)
    

    # Use a list comprehension to "pick" the items
    selected_indices_level_1 = [selected_indices_1[i] for i in sorted_indices[:level_1_budget].tolist()]
    selected_indices_level_2 = [selected_indices_1[i] for i in sorted_indices[level_1_budget:level_1_budget + level_2_budget].tolist()]
    selected_indices_level_3 = [selected_indices_1[i] for i in sorted_indices[level_1_budget + level_2_budget:level_1_budget + level_2_budget + level_3_budget].tolist()]
    
    # print(f"selected_indices_level_1: {selected_indices_level_1}", flush=True)
    # print(f"selected_indices_level_2: {selected_indices_level_2}", flush=True)
    # print(f"selected_indices_level_3: {selected_indices_level_3}", flush=True)
    # print(f"level_1_budget: {level_1_budget}", flush=True)
    # print(f"level_2_budget: {level_2_budget}", flush=True)
    # print(f"level_3_budget: {level_3_budget}", flush=True)
    
    time_indices = []
    new_raw_video = []
    for i in range(len(selected_indices_level_1)):
        new_raw_video.append(raw_video[selected_indices_level_1[i]])
        time_indices.append(selected_indices_level_1[i])
    for i in range(len(selected_indices_level_2)):
        # concat 4 frames to one hyperframe (2x2 grid)
        current_frame_idx = selected_indices_level_2[i]
        
        # We want 4 frames. Let's aim for 2 before and 2 after (or similar)
        lo = max(0, current_frame_idx - 1)
        hi = min(len(raw_video), current_frame_idx + 3)
        pos = list(range(lo, hi))
        
        # Pad if we are at the edges of the video to ensure exactly 4 frames
        while len(pos) < 4:
            if lo > 0:
                lo -= 1
                pos.insert(0, lo)
            elif hi < len(raw_video):
                pos.append(hi)
                hi += 1
            else:
                break
                
        pos = pos[:4] # Ensure exactly 4 frames
        frames = [raw_video[p] for p in pos]
        
        if isinstance(frames[0], torch.Tensor):
            # PyTorch tensors: dim 1 is Height, dim 2 is Width (usually C, H, W)
            row0 = torch.cat(frames[0:2], dim=2) # Concat horizontally
            row1 = torch.cat(frames[2:4], dim=2) # Concat horizontally
            hyperframe = torch.cat([row0, row1], dim=1) # Concat vertically
            hyperframe_np = hyperframe.permute(1, 2, 0).numpy()
            H, W, _ = hyperframe_np.shape
            hyperframe_np = cv2.resize(hyperframe_np, (W // 2, H // 2))
            hyperframe_np = hyperframe_np.transpose(2, 0, 1)
            hyperframe = torch.from_numpy(hyperframe_np)
            new_raw_video.append(hyperframe)
        else:
            # Numpy arrays: axis 0 is Height, axis 1 is Width
            row0 = np.concatenate([frames[0], frames[1]], axis=1) # Concat horizontally
            row1 = np.concatenate([frames[2], frames[3]], axis=1) # Concat horizontally
            hyperframe = np.concatenate([row0, row1], axis=0) # Concat vertically
            new_raw_video.append(hyperframe)
            
        time_indices.append(selected_indices_level_2[i])
        
    for i in range(len(selected_indices_level_3)):
        # concat 16 frames to one hyperframe (4x4 grid)
        current_frame_idx = selected_indices_level_3[i]
        
        # Define a window around the current index
        # (Starting with ~8 before and ~8 after)
        lo = max(0, current_frame_idx - 8)
        hi = min(len(raw_video), current_frame_idx + 8)
        pos = list(range(lo, hi))
        
        # Expand the window if we hit the boundaries to ensure exactly 16 frames
        while len(pos) < 16:
            if lo > 0:
                lo -= 1
                pos.insert(0, lo)
            elif hi < len(raw_video):
                pos.append(hi)
                hi += 1
            else:
                break
                
        pos = pos[:16]
        frames = [raw_video[p] for p in pos]
        
        if isinstance(frames[0], torch.Tensor):
            # Create 4 rows by concatenating 4 frames horizontally (dim=2)
            row0 = torch.cat(frames[0:4], dim=2)
            row1 = torch.cat(frames[4:8], dim=2)
            row2 = torch.cat(frames[8:12], dim=2)
            row3 = torch.cat(frames[12:16], dim=2)
            # Stack the 4 rows vertically (dim=1)
            hyperframe = torch.cat([row0, row1, row2, row3], dim=1)
            hyperframe_np = hyperframe.permute(1, 2, 0).numpy()
            H, W, _ = hyperframe_np.shape
            hyperframe_np = cv2.resize(hyperframe_np, (W // 4, H // 4))
            hyperframe_np = hyperframe_np.transpose(2, 0, 1)
            hyperframe = torch.from_numpy(hyperframe_np)
            new_raw_video.append(hyperframe)
        else:
            # Create 4 rows by concatenating 4 frames horizontally (axis=1)
            row0 = np.concatenate(frames[0:4], axis=1)
            row1 = np.concatenate(frames[4:8], axis=1)
            row2 = np.concatenate(frames[8:12], axis=1)
            row3 = np.concatenate(frames[12:16], axis=1)
            # Stack the 4 rows vertically (axis=0)
            hyperframe = np.concatenate([row0, row1, row2, row3], axis=0)
            new_raw_video.append(hyperframe)
            
        time_indices.append(selected_indices_level_3[i])

    # sort by time indices
    order = torch.argsort(torch.tensor(time_indices))
    time_indices = [time_indices[j] for j in order]
    new_raw_video = [new_raw_video[j] for j in order]
    
    # print(f"time_indices: {time_indices}", flush=True)

    video_inputs = new_raw_video
    # save the video_inputs for debugging
    # for i, frame in enumerate(video_inputs):
    #     cv2.imwrite(f"images/{video_name}_{i}.png", frame.permute(1, 2, 0).numpy())

    assert len(video_inputs) == args.num_frames

    
    
    if "llava_video" in args.model_name:
        video = lens.image_processor.preprocess(raw_video, return_tensors="pt")["pixel_values"].cuda().to(dtype=torch.bfloat16)
        video_inputs = [video]
    if type(video_inputs) is not list:
        video_inputs = [video_inputs]
    # except:
    #     continue


    subtitle_text_content = ""
    if subtitles:
        formatted_list = []
        for item in subtitles:
            if isinstance(item, tuple):
                # Handles (timestamp, text) from SRT/JSON
                timestamp, text = item
                formatted_list.append(f"[{timestamp}s]: {text}")
            else:
                # Handles list of strings from .txt files
                formatted_list.append(str(item))
        
        # Join into a clean block
        subtitle_text_content = "\n".join(formatted_list)

        # --- Final Prompt Assembly ---
        subtitle_header = "This video's subtitles are listed below for context:\n"
        # Construct the full prompt block
        full_prompt = (
            f"{subtitle_header}"
            f"{subtitle_text_content}\n\n"
            f"{prompt}\n"
            f"Respond with only the letter of the correct option.\n"
        )
    else:
        full_prompt = prompt + "\nRespond with only the letter of the correct option.\n"
    
    pred = lens.mllm_response(lens.video_llm, lens.processor, lens.image_processor, full_prompt, None, video_inputs, max_new_tokens=512)
    print(f"Video: {video_name}, Question: {question}, Prediction: {pred}, Answer: {answer}", flush=True)

    output.append(
        {
            "question": question,
            "candidates": candidates,
            "task_type": duration if args.task == "videomme" else task_type,
            "video_name": video_name,
            "duration": len(raw_video),
            "domain": line.get("domain", None),
            "sub_category": line.get("sub_category", None),
            "video_id": line.get("video_id", None),
            # "target_text": target_text,
            "pred": pred,
            "answer": answer,
        }
    )
    print(output[-1], flush=True)

    processed_identifiers.add(current_identifier)
    with open(checkpoint_file, 'w') as f:
        json.dump({'output': output, 'processed_identifiers': list(processed_identifiers)}, f, indent=4)
    
    print(f"Rank {world_rank} Output for {video_name[:8]}... - {question[:20]}...: {output[-1]['pred']}, answer: {answer}", flush=True)

dist.barrier()
    
final_output = [None] * world_size
dist.all_gather_object(
    final_output,
    output,
)
all_output = list(chain(*final_output))

global_rank = dist.get_rank()
if global_rank == 0:
    output_filename = os.path.join(checkpoint_dir, f"output.json")
    with open(output_filename, "w") as f:
        json.dump(all_output, f)
    
    result = {}
    task_types = set([item['task_type'] for item in all_output])
    for task_type in task_types:
        task_type_output = [item for item in all_output if item['task_type'] == task_type]
        accuracy = sum(1 for item in task_type_output if item['answer'] in item['pred'] or item['pred'] in item['answer']) / len(task_type_output)
        result[task_type] = accuracy
    result["overall"] = sum(1 for item in all_output if item['answer'] in item['pred'] or item['pred'] in item['answer']) / len(all_output)
    print(result)
    
    result_filename = os.path.join(checkpoint_dir, f"result.json")
    with open(result_filename, "w") as f:
        json.dump(result, f)

    for rank_idx in range(world_size):
        rank_checkpoint_file = os.path.join(checkpoint_dir, f"cuda:{rank_idx}.json")
        if os.path.exists(rank_checkpoint_file):
            os.remove(rank_checkpoint_file)