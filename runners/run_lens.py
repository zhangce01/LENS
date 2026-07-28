import os
import json
import datetime
from itertools import chain

import numpy as np
import cv2
import torch
from torch import distributed as dist
from tqdm import tqdm
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers import (
    CLIPProcessor, CLIPModel,
    Blip2ForImageTextRetrieval, Blip2Processor,
    BlipProcessor, BlipForImageTextRetrieval,
)

from utils.data import (
    EvalDatasetMLVU,
    EvalDatasetVideoMME,
    EvalDatasetLongVideoBench,
    EvalDatasetEgoschemaSubset,
    EvalDatasetNExTQA,
)
from utils.lens import LENS
from utils.sampling import (
    clip_based_sampling,
    blip_based_sampling,
    blip2_based_sampling,
    uniform_sampling,
    graph_based_sampling,
)
from utils.config import get_args
from utils.prompts import ALLOCATION_PROMPT
from API_CLIP.main import get_model as clip_mask_get_model, api_clip


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

# ITM model for query-frame relevance scoring
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
if args.itm_model_name == "BLIP2":  # blip2-itm-vit-g
    blip2_model = Blip2ForImageTextRetrieval.from_pretrained("Salesforce/blip2-itm-vit-g").to(device)
    blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-itm-vit-g")
if args.itm_model_name == "BLIP":
    blip_model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-large-coco").to(device)
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-large-coco")

# CLIP model for query-conditioned attention masking (spatial zoom-in)
clip_mask_model, clip_mask_prs, clip_mask_preprocess, _, clip_mask_tokenizer = clip_mask_get_model(
    model_name="ViT-L-14-336",
    layer_index=22,
    device=device
)


for line in pbar:
    video_name = line.get("video_name", None)
    answer = line.get("answer", None)
    prompt = line.get("prompt", None)
    question = line.get("question", None)
    task_type = line.get("task_type", None)
    video_path = line.get("video_path", None)
    candidates = line.get("candidates", None)
    duration = line.get("duration", None)

    current_identifier = (video_name, question)
    if current_identifier in processed_identifiers:
        continue

    if not os.path.exists(video_path):
        print(f"Missing video: {video_path}")
        continue
    try:
        raw_video, _, _, frame_idx, fps, video_inputs, size_list = lens.load_video(video_path, args)
        if type(raw_video) is list:
            raw_video = raw_video[0]

        # Prompt the LLM to dynamically allocate the frame budget between the
        # two sampling strategies: r for spatial zoom-in, (1 - r) for temporal zoom-out.
        allocation_prompt = ALLOCATION_PROMPT.format(USER_QUERY=question, TOTAL_BUDGET=args.num_frames)
        response = lens.mllm_response(lens.video_llm, lens.processor, lens.image_processor, allocation_prompt, None, None, max_new_tokens=16)
        try:
            r = float(response.strip())
        except ValueError:
            r = 0.5  # fallback if parsing fails
        r = max(0.0, min(1.0, r))

        level_1_budget = int(round(r * args.num_frames))
        level_2_budget = args.num_frames - level_1_budget

        # Stage 1 (spatial zoom-in): ITM-based selection with the LLM-decided budget
        if args.itm_model_name == "CLIP":
            selected_indices_1, scores = clip_based_sampling(clip_model, clip_processor, raw_video, prompt, level_1_budget)
        elif args.itm_model_name == "BLIP2":
            selected_indices_1, scores = blip2_based_sampling(blip2_model, blip2_processor, raw_video, prompt, level_1_budget)
        elif args.itm_model_name == "BLIP":
            selected_indices_1, scores = blip_based_sampling(blip_model, blip_processor, raw_video, prompt, level_1_budget)
        else:
            raise ValueError(f"Invalid ITM model name: {args.itm_model_name}")

        # Candidate pool for the temporal branch: uniformly pre-sample the video
        # down to a fixed size before building the SSIM graph. Never ask for more
        # frames than the video has, otherwise the pool contains duplicates.
        n_pre = min(args.pre_sample_frames, len(raw_video))
        pre_selected_indices = uniform_sampling(raw_video, question, n_pre)

        sampled_video = raw_video[pre_selected_indices]
        scores = scores[pre_selected_indices]

        # Stage 2 (temporal zoom-out): graph-diffusion refinement with the remaining budget
        selected_indices_2, scores_2, confidence_shift, tau, norm_entropy = graph_based_sampling(
            sampled_video, scores, args.num_frames, device, beta=0.8, num_iters=10, sigma=0.3, use_ssim=True
        )

        # Re-rank the graph-refined candidates by scores_2 and keep at most
        # level_2_budget of them, requiring a minimum temporal gap to the
        # stage-1 selections (after mapping back to original frame indices).
        MIN_FRAME_GAP = 1  # can be raised to enforce a larger temporal gap (e.g. 4 frames)

        if isinstance(selected_indices_2, list):
            sel2_tensor = torch.tensor(selected_indices_2, device=device, dtype=torch.long)
        else:
            sel2_tensor = selected_indices_2.to(device)
        scores2_tensor = scores_2.to(device)

        # anchor indices in descending order of refined score
        order2 = torch.argsort(scores2_tensor[sel2_tensor], descending=True)
        stage1_indices = [int(i) for i in selected_indices_1]  # original-video indices

        filtered_sel2 = []
        for idx in order2.tolist():
            if len(filtered_sel2) >= level_2_budget:
                break
            cand_idx = int(sel2_tensor[idx].item())           # index into sampled_video
            orig_idx = int(pre_selected_indices[cand_idx])    # mapped back to original video

            too_close = False
            for s1 in stage1_indices:
                if abs(s1 - orig_idx) < MIN_FRAME_GAP:
                    too_close = True
                    break
            if too_close:
                continue
            filtered_sel2.append(cand_idx)

        # Refill from the remaining candidates (ignoring the gap constraint)
        # if the filter left us short of the stage-2 budget.
        if len(filtered_sel2) < level_2_budget:
            for idx in order2.tolist():
                cand_idx = int(sel2_tensor[idx].item())
                if cand_idx not in filtered_sel2:
                    filtered_sel2.append(cand_idx)
                if len(filtered_sel2) >= level_2_budget:
                    break

        selected_indices_2 = filtered_sel2

        new_raw_video = []
        time_indices = []

        # Stage-1 frames: keep full resolution, mask query-irrelevant regions
        for i in range(len(selected_indices_1)):
            frame = raw_video[selected_indices_1[i]]
            masked_frame = api_clip(frame, prompt, model=clip_mask_model, prs=clip_mask_prs, preprocess=clip_mask_preprocess, device=device, tokenizer=clip_mask_tokenizer)
            new_raw_video.append(masked_frame)
            time_indices.append(selected_indices_1[i])

        # Stage-2 frames: compose each anchor with its temporal neighborhood
        # into a 2x2 hyperframe, downscaled back to the original frame size
        for i in range(len(selected_indices_2)):
            current_frame_idx = selected_indices_2[i]

            # window size = 4: use +-1 and pad
            lo = max(0, current_frame_idx - 1)
            hi = min(len(sampled_video), current_frame_idx + 3)
            pos = list(range(lo, hi))

            # pad to 4 by repeating boundary
            while len(pos) < 4:
                if lo > 0:
                    lo -= 1
                    pos.insert(0, lo)
                elif hi < len(sampled_video):
                    pos.append(hi)
                    hi += 1
                else:
                    break

            pos = pos[:4]
            frames = [sampled_video[p] for p in pos]

            if isinstance(frames[0], torch.Tensor):
                # 2x2 grid
                row0 = torch.cat(frames[0:2], dim=2)   # concat width
                row1 = torch.cat(frames[2:4], dim=2)
                hyperframe = torch.cat([row0, row1], dim=1)  # concat height

                hyperframe_np = hyperframe.permute(1, 2, 0).numpy()
                H, W, _ = hyperframe_np.shape
                hyperframe_np = cv2.resize(hyperframe_np, (W // 2, H // 2))
                hyperframe_np = hyperframe_np.transpose(2, 0, 1)

                hyperframe = torch.from_numpy(hyperframe_np)
                new_raw_video.append(hyperframe)
                time_indices.append(pre_selected_indices[current_frame_idx])

            else:
                # numpy (H,W,C)
                row0 = np.concatenate([frames[0], frames[1]], axis=1)
                row1 = np.concatenate([frames[2], frames[3]], axis=1)
                hyperframe = np.concatenate([row0, row1], axis=0)

                new_raw_video.append(hyperframe)
                time_indices.append(pre_selected_indices[current_frame_idx])

        # sort by time indices
        order = np.argsort(time_indices)
        time_indices = [int(time_indices[j]) for j in order]
        new_raw_video = [new_raw_video[j] for j in order]

        video_inputs = new_raw_video

        assert len(video_inputs) == args.num_frames

        if "llava_video" in args.model_name:
            video = lens.image_processor.preprocess(raw_video, return_tensors="pt")["pixel_values"].cuda().to(dtype=torch.bfloat16)
            video_inputs = [video]
        if type(video_inputs) is not list:
            video_inputs = [video_inputs]
    except Exception as e:
        print(f"Rank {world_rank}: skipping {video_name} ({question[:40]}...): {e}", flush=True)
        continue

    prompt = prompt + "Respond with only the letter of the correct option.\n"
    pred = lens.mllm_response(lens.video_llm, lens.processor, lens.image_processor, prompt, None, video_inputs, max_new_tokens=512)
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
            "spatial_zoom_in_budget": level_1_budget,
            "temporal_zoom_out_budget": level_2_budget,
            "time_indices": time_indices,
            "pred": pred,
            "answer": answer
        }
    )

    processed_identifiers.add(current_identifier)
    with open(checkpoint_file, 'w') as f:
        json.dump({'output': output, 'processed_identifiers': list(processed_identifiers)}, f, indent=4)

torch.distributed.barrier()

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
