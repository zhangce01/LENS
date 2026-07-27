# LENS: Query-Adaptive Spatial Zoom-In and Temporal Zoom-Out for Long Video Understanding

LENS is a training-free keyframe selection pipeline for long-video question answering with large vision-language models (LVLMs). Given a fixed frame budget, LENS lets the LLM itself decide — per question — how to split the budget between two complementary views of the video:

- **Spatial zoom-in**: full-resolution keyframes whose query-irrelevant regions are suppressed by CLIP attention masking, for detail questions (reading text, small objects, attributes).
- **Temporal zoom-out**: 2×2 *hyperframes* that pack an anchor frame with its temporal neighborhood into a single frame slot, for questions about events, ordering, and changes over time.

## Pipeline

```
Question ──► LLM budget allocator ──► r ∈ [0, 1]
                                       │
        ┌──────────────────────────────┴──────────────────────────────┐
        │  r · N frames                                (1 − r) · N frames
        ▼                                                             ▼
  Spatial zoom-in                                          Temporal zoom-out
  1. ITM scoring (CLIP / BLIP / BLIP2)                1. Uniform pre-sampling (128 frames)
  2. Watershed anchor selection                       2. Graph-diffusion score refinement
  3. CLIP attention masking                              (SSIM frame graph + manifold ranking)
     (highlight query-relevant regions)               3. Watershed anchor selection
                                                      4. 2×2 hyperframe composition
        └──────────────────────────────┬──────────────────────────────┘
                                       ▼
                     N frames, temporally ordered ──► LVLM ──► answer
```

Key components:

- **LLM budget allocation** (`utils/prompts.py`): a lightweight prompt asks the LVLM to output a single ratio `r` from the question alone — detail questions get more spatial zoom-in, temporal questions get more zoom-out.
- **Watershed anchor selection** (`utils/sampling.py`): valleys of the query-frame relevance curve split the video into basins; each basin contributes its peak, and 1D k-means over time enforces temporal diversity.
- **Graph-diffusion refinement** (`utils/sampling.py`): frames are connected by pairwise SSIM (most-similar edges first, until the graph is connected); manifold ranking `f ← βPf + (1−β)y` propagates relevance along the graph, suppressing isolated noisy peaks and strengthening coherent segments.
- **CLIP attention masking** (`API_CLIP/`): text-conditioned attention maps from a hooked CLIP ViT gray out regions irrelevant to the question, spending the frame's resolution on what matters.

## Repository structure

```
LENS/
├── runners/
│   ├── run_lens.py    # main method
│   ├── run_baseline.py# ITM retrieval baseline (top-scoring frames only)
│   ├── run_uniform.py # uniform sampling baseline
│   └── run_qframe.py  # fixed hierarchical budget baseline (1/2 raw, 1/4 2×2, 1/4 4×4)
├── utils/
│   ├── lens.py        # LVLM backend loader
│   ├── sampling.py    # scoring, SSIM graph, diffusion, watershed selection
│   ├── prompts.py     # LLM budget-allocation prompt
│   ├── data.py        # benchmark datasets and subtitle parsing
│   └── config.py      # command-line arguments
├── models/            # per-LVLM adapters (Qwen2-VL / Qwen2.5-VL, InternVL2.5, LLaVA-Video, LongVU)
├── API_CLIP/          # CLIP attention masking (adapted from API prompting / clip_prs)
└── scripts/           # torchrun launch scripts
```

## Setup

```bash
conda create -n lens python=3.11
conda activate lens
pip install -r requirements.txt
# flash-attn is required by the Qwen-VL backend; install a wheel matching your CUDA/torch:
pip install flash-attn --no-build-isolation
```

The default backend is Qwen2.5-VL-7B-Instruct; CLIP ViT-L/14-336 is used for frame scoring and masking. All models are downloaded automatically from Hugging Face on first run.

Optional backends need their own packages: LLaVA-Video (`llava`), LongVU (`longvu`).

## Data preparation

Download the benchmarks and arrange them as the loaders in `utils/data.py` expect:

| Task flag | Benchmark | Expected layout under `--data_path` |
|---|---|---|
| `mlvu` | [MLVU](https://huggingface.co/datasets/MLVU/MVLU) | `json/*.json`, `video/<subtask>/` |
| `videomme` | [Video-MME](https://huggingface.co/datasets/lmms-lab/Video-MME) | `videomme/test-00000-of-00001.parquet`, `data/*.mp4`, `subtitle/*.srt` |
| `lvb` | [LongVideoBench](https://huggingface.co/datasets/longvideobench/LongVideoBench) | `lvb_val.json`, `videos/`, `subtitles/` |
| `egoschema` | [EgoSchema](https://huggingface.co/datasets/lmms-lab/egoschema) | `Subset/test-00000-of-00001.parquet`, `videos/*.mp4` |
| `nextqa` | [NExT-QA](https://huggingface.co/datasets/lmms-lab/NExTQA) | `MC/test-00000-of-00001.parquet`, `NExTVideo/` |

## Running

```bash
torchrun \
  --nnodes=1 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port=29501 \
  --nproc_per_node=4 \
  -m runners.run_lens \
  --itm_model_name CLIP \
  --model_name qwenvl25_7b \
  --task videomme \
  --data_path /path/to/Video-MME \
  --num_frames 32
```

See `scripts/` for ready-to-edit launch scripts of LENS and all baselines (`run_uniform`, `run_baseline`, `run_qframe`).

Main arguments (`utils/config.py`):

| Argument | Default | Description |
|---|---|---|
| `--model_name` | `qwenvl25_7b` | LVLM backend key (see `utils/lens.py: MODEL_MAP`) |
| `--task` | `mlvu` | `mlvu` / `videomme` / `lvb` / `egoschema` / `nextqa` |
| `--data_path` | `./data` | benchmark root directory |
| `--num_frames` | `32` | total frame budget N |
| `--itm_model_name` | `CLIP` | relevance scorer: `CLIP` / `BLIP` / `BLIP2` |
| `--fps` | `1.0` | decoding frame rate |
| `--output_path` | `./eval` | results directory |

Evaluation is distributed across GPUs with per-rank JSON checkpoints, so an interrupted run resumes automatically. Rank 0 writes `output.json` (all predictions) and `result.json` (accuracy per question type and overall) to `<output_path>/<model_name>/<task>/`.

## Using your own LVLM

Add a module under `models/` exposing three functions, then register it in `utils/lens.py: MODEL_MAP`:

```python
def load_model(model_path):   # -> (processor, video_llm, image_processor, _)
def load_video(video_path, args)
def mllm_response(video_llm, processor, image_processor, text, image_inputs, video, ...)
```

## Acknowledgements

This codebase builds on the evaluation frameworks of [AKS](https://github.com/ncTimTang/AKS) and [Vgent](https://github.com/xiaoqian-shen/Vgent). The CLIP attention masking in `API_CLIP/` is adapted from [API prompting](https://github.com/yu-rp/apiprompting) and [clip_prs](https://github.com/yossigandelsman/clip_prs); video loading utilities in `models/utils.py` are adapted from [qwen-vl-utils](https://github.com/QwenLM/Qwen2.5-VL). We thank the authors of these projects.

## License

This project is released under the [MIT License](LICENSE). `API_CLIP/clip_prs/` retains the license of its upstream project.

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{lens2026,
  title={LENS: Query-Adaptive Spatial Zoom-In and Temporal Zoom-Out for Long Video Understanding},
  author={},
  year={2026}
}
```
