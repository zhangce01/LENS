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
    

    # CLIP-BASED SELECTION
    if args.itm_model_name == "CLIP":
        selected_indices_1, scores = clip_based_sampling(clip_model, clip_processor, raw_video, prompt, args.num_frames)
    elif args.itm_model_name == "BLIP2":
        selected_indices_1, scores = blip2_based_sampling(blip2_model, blip2_processor, raw_video, question, args.num_frames)
    elif args.itm_model_name == "BLIP":
        selected_indices_1, scores = blip_based_sampling(blip_model, blip_processor, raw_video, question, args.num_frames)
    else:
        raise ValueError(f"Invalid ITM model name: {args.itm_model_name}")
    
    new_raw_video = raw_video[selected_indices_1]

    video_inputs = new_raw_video

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