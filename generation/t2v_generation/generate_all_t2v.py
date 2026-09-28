"""AIME - batch video generation with all 4 free/open-weight T2V models.

Runs Wan2.1-T2V-1.3B, CogVideoX-2B, LTX-Video-0.9.7-distilled and HunyuanVideo
sequentially, one prompt at a time, for the 57 video-compatible AIME prompts
(Modality in {'Video', 'Both'} out of the 100-prompt revision sheet).

All 4 transformers are loaded 4-bit quantized (bitsandbytes NF4) to keep VRAM
usage as low as possible regardless of the GPU this runs on. Models are
processed one at a time and fully unloaded before the next one loads.

Usage:
    python generate_all_t2v.py                     # all 4 models, 1 video/prompt
    python generate_all_t2v.py --models wan21 ltxvideo
    python generate_all_t2v.py --videos-per-prompt 2
    python generate_all_t2v.py --smoke-test         # 1 prompt, short clips, sanity check

Requires HF_TOKEN in a .env file at the repo root (same variable used by the
other generation scripts in this repo).
"""
import argparse
import datetime
import gc
import hashlib
import os

import pandas as pd
import torch
from diffusers.utils import export_to_video
from diffusers.quantizers import PipelineQuantizationConfig
from dotenv import load_dotenv
from huggingface_hub import login

MODEL_MAP = {
    'wan21':        ('Wan-AI/Wan2.1-T2V-1.3B-Diffusers', 'Wan2.1_T2V_1.3B'),
    'cogvideox':    ('zai-org/CogVideoX-2b', 'CogVideoX_2B'),
    'ltxvideo':     ('Lightricks/LTX-Video-0.9.7-distilled', 'LTXVideo_0.9.7_distilled'),
    'hunyuanvideo': ('hunyuanvideo-community/HunyuanVideo', 'HunyuanVideo'),
}

GEN_PARAMS = {
    'wan21': dict(
        height=480, width=832, num_frames=81, num_inference_steps=30,
        guidance_scale=5.0, flow_shift=3.0, fps=16,
    ),
    'cogvideox': dict(
        height=480, width=720, num_frames=49, num_inference_steps=50,
        guidance_scale=6.0, fps=8,
    ),
    'ltxvideo': dict(
        height=480, width=704, num_frames=121, num_inference_steps=8,
        guidance_scale=1.0, fps=24,  # distilled checkpoint: guidance_scale MUST be 1.0
    ),
    'hunyuanvideo': dict(
        height=320, width=512, num_frames=61, num_inference_steps=30,
        guidance_scale=6.0, fps=15,
    ),
}

LOW_QUALITY_NEGATIVE = 'worst quality, inconsistent motion, blurry, jittery, distorted'
SMOKE_TEST_FRAMES = 33  # short clip (4*8+1), valid frame count for all 4 VAEs


def quant_config():
    return PipelineQuantizationConfig(
        quant_backend='bitsandbytes_4bit',
        quant_kwargs={'load_in_4bit': True, 'bnb_4bit_quant_type': 'nf4', 'bnb_4bit_compute_dtype': torch.bfloat16},
        components_to_quantize=['transformer'],
    )


def load_pipeline(model_key, hf_token):
    hf_model_id, _ = MODEL_MAP[model_key]
    qconfig = quant_config()

    if model_key == 'wan21':
        from diffusers import AutoencoderKLWan, WanPipeline
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

        vae = AutoencoderKLWan.from_pretrained(hf_model_id, subfolder='vae', torch_dtype=torch.float32, token=hf_token)
        pipe = WanPipeline.from_pretrained(
            hf_model_id, vae=vae, quantization_config=qconfig, torch_dtype=torch.bfloat16, token=hf_token,
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=GEN_PARAMS['wan21']['flow_shift'],
        )

    elif model_key == 'cogvideox':
        from diffusers import CogVideoXPipeline

        pipe = CogVideoXPipeline.from_pretrained(
            hf_model_id, quantization_config=qconfig, torch_dtype=torch.bfloat16, token=hf_token,
        )

    elif model_key == 'ltxvideo':
        from diffusers import LTXPipeline

        pipe = LTXPipeline.from_pretrained(
            hf_model_id, quantization_config=qconfig, torch_dtype=torch.bfloat16, token=hf_token,
        )

    elif model_key == 'hunyuanvideo':
        from diffusers import HunyuanVideoPipeline

        pipe = HunyuanVideoPipeline.from_pretrained(
            hf_model_id, quantization_config=qconfig, torch_dtype=torch.bfloat16, token=hf_token,
        )

    else:
        raise ValueError(f'Unknown model key: {model_key}')

    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def make_filename(prompt_id, prompt_type, index=0):
    ts = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
    h = hashlib.sha256(f'{prompt_id}_{index}_{ts}'.encode()).hexdigest()[:10]
    t = 'amb' if prompt_type == 'Ambiguous' else 'exp'
    return f'P{prompt_id:03d}_{t}_{h}.mp4'


def generate_video(pipe, model_key, prompt_text, num_frames):
    params = GEN_PARAMS[model_key]
    with torch.no_grad():
        if model_key == 'ltxvideo':
            result = pipe(
                prompt=prompt_text,
                negative_prompt=LOW_QUALITY_NEGATIVE,
                height=params['height'], width=params['width'],
                num_frames=num_frames,
                num_inference_steps=params['num_inference_steps'],
                guidance_scale=params['guidance_scale'],
            )
        else:
            result = pipe(
                prompt=prompt_text,
                height=params['height'], width=params['width'],
                num_frames=num_frames,
                num_inference_steps=params['num_inference_steps'],
                guidance_scale=params['guidance_scale'],
            )
    return result.frames[0]


def run_model(model_key, df_vid, repo_root, hf_token, videos_per_prompt, smoke_test, index_suffix):
    hf_model_id, model_label = MODEL_MAP[model_key]
    output_dir = os.path.join(repo_root, 'dataset', 'media', 'videos', model_label)
    index_name = f'generated_{model_label}.csv' if not index_suffix else f'generated_{model_label}_{index_suffix}.csv'
    index_csv = os.path.join(repo_root, 'dataset', 'labels', index_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(index_csv), exist_ok=True)

    if os.path.exists(index_csv):
        df_done = pd.read_csv(index_csv)
        done_ids = set(zip(df_done['prompt_type'], df_done['prompt_id'], df_done['vid_index']))
        print(f'[{model_label}] Resuming: {len(df_done)} videos already generated.')
    else:
        df_done = pd.DataFrame()
        done_ids = set()
        print(f'[{model_label}] Starting fresh.')

    print(f'[{model_label}] Loading {hf_model_id} (4-bit quantized transformer)...')
    pipe = load_pipeline(model_key, hf_token)
    print(f'[{model_label}] Loaded.')

    num_frames = SMOKE_TEST_FRAMES if smoke_test else GEN_PARAMS[model_key]['num_frames']
    records = []
    df_out = df_done

    for _, row in df_vid.iterrows():
        pid = int(row['ID'])
        ptype = row['Type']
        prompt_text = str(row['Prompt Text'])

        for vid_idx in range(videos_per_prompt):
            if (ptype, pid, vid_idx) in done_ids:
                continue

            status, fname = 'ok', ''
            try:
                frames = generate_video(pipe, model_key, prompt_text, num_frames)
                fname = make_filename(pid, ptype, vid_idx)
                export_to_video(frames, os.path.join(output_dir, fname), fps=GEN_PARAMS[model_key]['fps'])
                done_ids.add((ptype, pid, vid_idx))
                torch.cuda.empty_cache()
                gc.collect()
            except torch.cuda.OutOfMemoryError as e:
                print(f'  [{model_label}] OOM P{pid:03d}: {str(e)[:150]}')
                status = 'oom_error'
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                print(f'  [{model_label}] Error P{pid:03d}: {str(e)[:150]}')
                status = 'error'

            records.append({
                'file_path': f'videos/{model_label}/{fname}' if fname else '',
                'prompt_id': pid,
                'vid_index': vid_idx,
                'prompt_type': ptype,
                'user_profile': row.get('User Profile', ''),
                'modality': row.get('Modality', 'Video'),
                'category': row.get('Category', ''),
                'language': row.get('Language', ''),
                'status': status,
                'model': model_label,
                'prompt_text': prompt_text,
            })

            # Persist after every video so a crash/kill mid-run loses at most one entry,
            # instead of losing the index for every video generated since the model loaded.
            df_new = pd.DataFrame(records)
            df_out = pd.concat([df_done, df_new], ignore_index=True) if not df_done.empty else df_new
            df_out.to_csv(index_csv, index=False)

    ok = (df_out['status'] == 'ok').sum() if not df_out.empty else 0
    total = len(df_out) if not df_out.empty else 0
    print(f'[{model_label}] Done. Saved: {ok} | Total rows: {total} | Index: {index_csv}')

    del pipe
    torch.cuda.empty_cache()
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description='AIME batch T2V generation (all free/open-weight models).')
    parser.add_argument('--models', nargs='+', choices=list(MODEL_MAP.keys()), default=list(MODEL_MAP.keys()),
                         help='Which models to run (default: all 4, in order).')
    parser.add_argument('--videos-per-prompt', type=int, default=1,
                         help='How many videos to generate per prompt per model (default: 1).')
    parser.add_argument('--smoke-test', action='store_true',
                         help='Restrict to 1 prompt and short clips, to sanity-check the pipeline end-to-end.')
    parser.add_argument('--repo-root', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')),
                         help='AIME-public repo root (default: two levels up from this script).')
    parser.add_argument('--prompts-file', default=os.path.join('prompts', 'all_prompts.xlsx'),
                         help='Prompts xlsx, relative to --repo-root (default: prompts/all_prompts.xlsx). '
                              'Expected columns: ID, Type, Prompt Text, User Profile, Modality, Category, '
                              'Language - Modality is optional (if absent, every row is treated as '
                              'video-compatible).')
    parser.add_argument('--sheet-name', default='All Prompts (100)',
                         help="Excel sheet name to read (default: 'All Prompts (100)').")
    parser.add_argument('--index-suffix', default='',
                         help='If set, writes to generated_{ModelLabel}_{suffix}.csv instead of '
                              'generated_{ModelLabel}.csv, so a new prompt batch never collides with (or '
                              'resumes into) an existing index - use this for any batch that is not the '
                              'original 100-prompt revision set, e.g. --index-suffix llm_generated.')
    args = parser.parse_args()

    load_dotenv(os.path.join(args.repo_root, '.env'))
    hf_token = os.environ.get('HF_TOKEN')
    if hf_token:
        login(hf_token)
    else:
        print('WARNING: HF_TOKEN not found in .env — gated repos will fail to download.')

    prompts_xlsx = os.path.join(args.repo_root, args.prompts_file)
    df_all = pd.read_excel(prompts_xlsx, sheet_name=args.sheet_name)
    df_all.columns = df_all.columns.str.strip()
    if 'Modality' in df_all.columns:
        df_vid = df_all[df_all['Modality'].isin(['Video', 'Both'])].copy().reset_index(drop=True)
    else:
        print("WARNING: no 'Modality' column found - treating every row as video-compatible.")
        df_vid = df_all.copy().reset_index(drop=True)

    if args.smoke_test:
        df_vid = df_vid.head(1).copy()

    print(f'Prompts file: {prompts_xlsx} (sheet: {args.sheet_name})')
    print(f'Video-compatible prompts: {len(df_vid)}')
    print(f'Models to run: {args.models}')
    print(f'Videos per prompt: {args.videos_per_prompt}')
    print(f'Index suffix: {args.index_suffix or "(none - default generated_{Model}.csv)"}')
    print(f'Smoke test: {args.smoke_test}')

    for model_key in args.models:
        run_model(model_key, df_vid, args.repo_root, hf_token, args.videos_per_prompt, args.smoke_test,
                  args.index_suffix)

    print('\nAll requested models finished.')


if __name__ == '__main__':
    main()
