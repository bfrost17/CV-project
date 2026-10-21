# DrVideo: Document Retrieval Based Long Video Understanding

> **CVPR 2025** — Code for the paper *“DrVideo: Document Retrieval Based Long Video Understanding”*.

Most of the existing methods for video understanding primarily focus on videos only lasting tens of seconds, with limited exploration of techniques for handling long videos. The increased number of frames in long videos poses two main challenges: difficulty in locating key information and performing long-range reasoning. Thus, we propose DrVideo, a document-retrieval-based system designed for long video understanding. Our key idea is to convert the long-video understanding problem into a long-document understanding task so as to effectively leverage the power of large language models. Specifically, DrVideo first transforms a long video into a coarse text-based long document to initially retrieve key frames and then updates the documents with the augmented key frame information. It then employs an agent-based iterative loop to continuously search for missing information and augment the document until sufficient question-related information is gathered for making the final predictions in a chain-of-thought manner. Extensive experiments on long video benchmarks confirm the effectiveness of our method. DrVideo significantly outperforms existing LLM-based state-of-the-art methods on EgoSchema benchmark (3 minutes), MovieChat-1K benchmark (10 minutes), and the long split of Video-MME benchmark (average of 44 minutes).

---

## Repository Structure
```
.
├── main.py                 # EgoSchema retrieval, frame augmentation, and QA
├── eval.py                 # EgoSchema evaluation and other benchmark utilities
├── dataset.py              # EgoSchema dataset loader
├── model.py                # OpenAI chat model wrapper
├── prompts.py              # Agent and answer prompts
├── models/
│   ├── blip2_model.py      # BLIP frame captioning used by the pipeline
│   └── clip_model.py       # CLIP image feature extraction helper
├── util.py                 # Argument parsing and JSON helpers
├── data/egoschema/         # Annotations, precomputed LaViLa captions, and metadata
├── dataset/                # Downloaded <uid>.mp4 files (git-ignored)
└── requirements.txt        # Pipeline dependencies; install CUDA PyTorch separately
```

---

## Installation
Install a CUDA-enabled PyTorch build appropriate for your GPU and Python version, then install the pipeline dependencies:

```bash
pip install -r requirements.txt
```

---

## Data Preparation
The default dataset interface follows **EgoSchema-style** multiple-choice QA.

Place (or symlink) files under `data/egoschema/`:
- `lavila_subset_add_time.json` — narrations / descriptions per `uid`
- `subset_anno.json` — question & options per `uid` (with optional `truth`)
- `duration.json` — video duration (seconds) per `uid`

The current dataset loader expects videos at:
```
dataset/{uid}.mp4
```
The initial LaViLa captions are precomputed and loaded by the pipeline; video-to-document caption generation is not run at inference time.

---

## Quick Start
Run the EgoSchema QA pipeline using the precomputed caption documents:

```bash
python main.py --num_examples_to_run -1 --output_base_path output --output_filename egoschema_run.json --start_from_scratch
```

---

## Evaluation Only
If you already have a predictions JSON (same schema the pipeline writes), you can compute MCQ accuracy:

```bash
python -c "from eval import eval_qa_egoschema; import json;  data=json.load(open('output/egoschema_run.json'));  acc=eval_qa_egoschema(data); print(acc)"
```

---

## Logging & Outputs
- Intermediate and final artifacts are written under `--output_base_path`.
- The main output file contains per-UID entries:
  ```json
  {
    "<uid>": {
      "pred": 0–4,            // model choice (A–E)
      "pred_text": "A–E",     // textual label
      "rationale": "...",      // (if enabled) model rationale
      "evidence": {...}        // captions / retrieved snippets
    },
    ...
  }
  ```

---

## Troubleshooting
- **CUDA OOM**: set `--image_captioner_device cpu` to run BLIP on the CPU.
- **OpenAI errors**: ensure the `--openai_api_key` is valid.

---

## Citation
If you find this repo useful, please cite:
```
@InProceedings{Ma_2025_CVPR,
    author    = {Ma, Ziyu and Gou, Chenhui and Shi, Hengcan and Sun, Bin and Li, Shutao and Rezatofighi, Hamid and Cai, Jianfei},
    title     = {DrVideo: Document Retrieval Based Long Video Understanding},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {18936-18946}
}
```

---

## License
This code is released for research purposes. Please check each model's own license (OpenAI, BLIP, CLIP, etc.).
