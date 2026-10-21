# DrVideo EgoSchema Trial

This repository runs an EgoSchema multiple-choice video QA pipeline based on DrVideo. It retrieves relevant timestamps from precomputed LaViLa captions, augments selected video frames with BLIP-base, uses a GPT judge/find loop to gather more evidence when needed, and asks GPT-4.1-mini for the final answer.

## Requirements

- Python environment with the packages in `requirements.txt`
- CUDA-enabled PyTorch installation for GPU-based BLIP inference (or run BLIP on CPU)
- OpenAI API key with API access and billing enabled
- EgoSchema subset data and matching videos

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

Place videos at `dataset\<uid>.mp4`. The `dataset` directory is git-ignored; obtain the dataset separately. The initial LaViLa video-to-document inference is not run by this pipeline: its captions are loaded from the provided files.

Set your API key in the same CMD window before running:

```cmd
set "OPENAI_API_KEY=your-api-key"
```

## Run

Run one question:

```cmd
python main.py --num_examples_to_run 1 --output_base_path output --output_filename single_trial.json --start_from_scratch
```

Run all questions in the configured subset:

```cmd
python main.py --num_examples_to_run -1 --output_base_path output --output_filename egoschema_run.json --start_from_scratch
```

The default chat model is `gpt-4.1-mini`, temperature is 0, and initial retrieval uses K=20. The run reports the baseline accuracy when it finishes.

## Paper

DrVideo: Document Retrieval Based Long Video Understanding, CVPR 2025. The local trial uses precomputed LaViLa captions, BLIP-base, and GPT-4.1-mini, so it is not an exact reproduction of the paper's full setup.
