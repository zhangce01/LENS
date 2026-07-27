import os
import json
import datetime
from itertools import chain

import torch
from torch import distributed as dist
from tqdm import tqdm
from transformers.trainer_pt_utils import IterableDatasetShard

from utils.data import (
    EvalDatasetMLVU,
    EvalDatasetVideoMME,
    EvalDatasetLongVideoBench,
    EvalDatasetEgoschemaSubset,
    EvalDatasetNExTQA,
)
from utils.lens import LENS
from utils.sampling import uniform_sampling
from utils.config import get_args


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

    raw_video, _, _, frame_idx, fps, video_inputs, size_list = lens.load_video(video_path, args)
    if type(raw_video) is list:
        raw_video = raw_video[0]

    selected_indices = uniform_sampling(raw_video, question, args.num_frames)
    sampled_video = raw_video[selected_indices]
    video_inputs = sampled_video

    if "llava_video" in args.model_name:
        video = lens.image_processor.preprocess(raw_video, return_tensors="pt")["pixel_values"].cuda().to(dtype=torch.bfloat16)
        video_inputs = [video]
    if type(video_inputs) is not list:
        video_inputs = [video_inputs]

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
