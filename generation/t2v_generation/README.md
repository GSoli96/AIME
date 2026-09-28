# AIME T2V batch generation

Standalone script that runs all 4 free/open-weight T2V models (Wan2.1-T2V-1.3B,
CogVideoX-2B, LTX-Video-0.9.7-distilled, HunyuanVideo) over the 57 video-compatible
AIME prompts, one model at a time, 4-bit quantized.

## What to copy to the other machine

This folder, plus (relative to the `AIME-public` repo root):
```
prompts/all_prompts.xlsx               # or prompts/llm_generated_prompts.xlsx for a new batch
.env               # must contain HF_TOKEN with read access to the gated repos used
```
Keep the same relative layout (`AIME-public/generation/t2v_generation/generate_all_t2v.py`,
`AIME-public/prompts/...`, `AIME-public/.env`) — the script locates the repo root as two
levels up from itself and reads `prompts/all_prompts.xlsx` / `.env` from there. Use
`--repo-root` to override if the layout differs.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python generate_all_t2v.py                       # all 4 models, 1 video/prompt, 57 prompts each
python generate_all_t2v.py --models wan21 ltxvideo
python generate_all_t2v.py --videos-per-prompt 2
python generate_all_t2v.py --smoke-test           # 1 prompt, short clips — sanity check first
```

### Running a different/new prompt batch (e.g. a follow-up study)

```bash
python generate_all_t2v.py --prompts-file prompts/llm_generated_prompts.xlsx \
    --sheet-name "Sheet1" --index-suffix llm_generated
```

- `--prompts-file`: xlsx path relative to `--repo-root`. Expected columns: `ID`, `Type`
  (`Ambiguous`/`Explicit`), `Prompt Text`. Optional: `Modality` (filters to
  `Video`/`Both` if present, otherwise every row is treated as video-compatible),
  `User Profile`, `Category`, `Language` (blank if missing).
- `--sheet-name`: Excel sheet to read (default `"All Prompts (100)"` — check the real
  sheet name in the new file, e.g. via `python -c "import pandas as pd; print(pd.ExcelFile('prompts/llm_generated_prompts.xlsx').sheet_names)"`).
- `--index-suffix`: **always set this for any batch that isn't the original 100-prompt
  revision set.** Without it, the new run's resumability check reads/writes the *same*
  `generated_{ModelLabel}.csv` as the original 57-prompt batch, which would misinterpret
  new prompts as "already done" if their numeric `ID` happens to coincide with an existing
  one, and would mix new/old rows into the same index. With `--index-suffix
  llm_generated`, output goes to `generated_{ModelLabel}_llm_generated.csv` instead —
  fully separate from the existing 228-video batch until a deliberate decision is made to
  merge them.

## Output

```
dataset/media/videos/{ModelLabel}/P{id}_{amb|exp}_{hash}.mp4
dataset/labels/generated_{ModelLabel}.csv                    # or generated_{ModelLabel}_{suffix}.csv
```

Each model's CSV index is resumable: re-running the script skips
`(prompt_type, prompt_id, vid_index)` combinations already marked `ok`/present in the
index, so an interrupted run can just be restarted.
