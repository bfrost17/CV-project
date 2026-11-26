# Experiment on DrVideo for CSI 5341 Learning Based Computer Vision

This project evaluates two changes to an EgoSchema multiple-choice video question-answering pipeline based on DrVideo:

1. **Hypothesis 1 — Multimodal retrieval:** combine question-to-caption text similarity with CLIP question-to-frame visual similarity when selecting initial frames.
2. **Hypothesis 2 — Confidence-guided augmentation:** use an LLM to assess retrieved captions and apply BLIP only to frames whose captions appear insufficient.

The experiments use precomputed LaViLa captions, BLIP-base for image captioning, and GPT-4.1-mini for assessment and reasoning. The initial LaViLa video-to-document inference is not run by this pipeline. This local trial is not an exact reproduction of the full DrVideo paper setup.

## Requirements

- Python environment with the packages in `requirements.txt`
- CUDA-enabled PyTorch installation for GPU-based BLIP and CLIP inference, or configure the models to run on CPU
- OpenAI API key with API access and billing enabled
- EgoSchema metadata, caption database, and matching videos

Install the dependencies:

```cmd
pip install -r requirements.txt
```

Install a PyTorch build appropriate for your Python version and GPU separately.

## Data

The default metadata and precomputed captions are read from `data\egoschema`:

- `subset_anno.json` — 500 questions, choices, and answers
- `lavila_subset_add_time.json` — timestamped precomputed LaViLa captions
- `duration.json` — video durations
- `database\<uid>.json` — per-video caption documents used for retrieval

Place videos at `dataset\<uid>.mp4`. The `dataset` directory is git-ignored; obtain the dataset separately.

Set the API key in the same CMD window before running:

```cmd
set "OPENAI_API_KEY=your-api-key"
```

## Completed experiments

All runs used the 500-question EgoSchema subset. The text-only baseline retrieves captions by question-to-caption similarity and applies BLIP to every retrieved frame. Hypothesis 1 uses text/visual score fusion with equal weights. Hypothesis 2 keeps text-only retrieval and skips initial BLIP for captions with sufficiency scores of at least 0.8. The combined run uses both changes.

| Run | Accuracy | OpenAI cost | Tokens | Requests | Processing errors |
|---|---:|---:|---:|---:|---:|
| CV baseline: text retrieval, BLIP on all retrieved frames | 66.4% | $2.37 | 5,787,178 | 1,726 | 1 |
| Hypothesis 1: multimodal retrieval, BLIP on all retrieved frames | 65.0% | $1.90 | 4,583,258 | 1,337 | 1 |
| Hypothesis 2: text retrieval, confidence-guided BLIP | 65.6% | $2.07 | 4,899,880 | 1,912 | 5 |
| Both hypotheses: multimodal retrieval and confidence-guided BLIP | 65.4% | $2.12 | 4,985,101 | 1,852 | 0 |

In the combined run, confidence-guided augmentation reduced the initial BLIP frame count by about 10.4% on average (17.93 frames captioned out of 20 retrieved). The results show lower OpenAI cost and token use than baseline for all three variants, but none improved the reported accuracy. Requests decreased for Hypothesis 1 and increased for Hypothesis 2 and the combined run.

## Reproducing the runs

Run commands from the repository root. Each output filename is separate, and `--start_from_scratch` prevents resuming an existing run.

### CV baseline

Text-only Top-20 retrieval, followed by BLIP on all retrieved frames:

```cmd
python main.py --num_examples_to_run -1 --retrieval_mode text_only --retrieval_top_k 20 --augmentation_mode all --output_base_path output --output_filename egoschema_500_baseline.json --start_from_scratch
```

### Hypothesis 1 — multimodal retrieval

Fuse text and CLIP visual similarity using a 0.5 visual weight; retain baseline augmentation:

```cmd
python main.py --num_examples_to_run -1 --retrieval_mode multimodal --retrieval_top_k 20 --retrieval_visual_weight 0.5 --augmentation_mode all --output_base_path output --output_filename egoschema_500_multimodal.json --start_from_scratch
```

### Hypothesis 2 — confidence-guided augmentation

Keep retrieval text-only and skip initial BLIP for captions scored at least 0.8:

```cmd
python main.py --num_examples_to_run -1 --retrieval_mode text_only --retrieval_top_k 20 --augmentation_mode confidence_guided --caption_sufficiency_threshold 0.8 --output_base_path output --output_filename egoschema_500_confidence_guided.json --start_from_scratch
```

### Combined experiment

Enable both multimodal retrieval and confidence-guided augmentation:

```cmd
python main.py --num_examples_to_run -1 --retrieval_mode multimodal --retrieval_top_k 20 --retrieval_visual_weight 0.5 --augmentation_mode confidence_guided --caption_sufficiency_threshold 0.8 --output_base_path output --output_filename egoschema_500_multimodal_confidence_guided.json --start_from_scratch
```

## Run parameters

- `--num_examples_to_run -1` runs all examples in the configured subset.
- `--retrieval_mode text_only|multimodal` selects text-only or fused retrieval. Text-only is the default.
- `--retrieval_top_k 20` sets the maximum number of initial retrieved frames.
- `--retrieval_visual_weight 0.5` gives text and visual similarity equal weight; it applies only to multimodal retrieval.
- `--augmentation_mode all|confidence_guided` selects baseline augmentation or confidence-guided augmentation. `all` is the default.
- `--caption_sufficiency_threshold 0.8` sets the confidence threshold at or above which initial BLIP augmentation is skipped.
- `--output_base_path` and `--output_filename` set the result location and filename.
- `--start_from_scratch` starts a fresh run rather than resuming from an existing output file.

The default chat model is `gpt-4.1-mini` with temperature 0. BLIP and CLIP devices default to CUDA.

## Paper

DrVideo: Document Retrieval Based Long Video Understanding, CVPR 2025.
