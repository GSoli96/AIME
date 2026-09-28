# ── Configuration ────────────────────────────────────────────────────────────
import os
from dotenv import load_dotenv
load_dotenv()

# Choose one: 'large' | 'medium' | 'turbo'
MODEL_KEY = 'large'

MODEL_MAP = {
    'large'  : ('stabilityai/stable-diffusion-3.5-large',       'SD35_Large'),
    'medium' : ('stabilityai/stable-diffusion-3.5-medium',      'SD35_Medium'),
    'turbo'  : ('stabilityai/stable-diffusion-3.5-large-turbo', 'SD35_Turbo'),
}

HF_MODEL_ID, MODEL_LABEL = MODEL_MAP[MODEL_KEY]
HF_TOKEN = os.environ.get('HF_TOKEN')  # needs read access to stabilityai models

IMAGES_PER_PROMPT    = 1
NUM_INFERENCE_STEPS  = 28   # 28 for large/medium; 4 for turbo
GUIDANCE_SCALE       = 7.5  # ignored by turbo (CFG=0)
BATCH_SIZE           = 1    # increase if VRAM allows (3060 = 12GB)

REPO_ROOT  = os.path.abspath(os.path.join(os.getcwd(), '..'))
PROMPTS_XLS = os.path.join(REPO_ROOT, 'prompts', 'all_prompts.xlsx')
OUTPUT_DIR  = os.path.join(REPO_ROOT, 'dataset', 'media', 'images', MODEL_LABEL)
INDEX_CSV   = os.path.join(REPO_ROOT, 'dataset', 'labels', f'generated_{MODEL_LABEL}.csv')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(INDEX_CSV), exist_ok=True)

print(f'Model   : {HF_MODEL_ID}')
print(f'Label   : {MODEL_LABEL}')
print(f'Output  : {OUTPUT_DIR}')

# ── Load prompts ─────────────────────────────────────────────────────────────
import pandas as pd

df_all = pd.read_excel(PROMPTS_XLS, sheet_name='All Prompts (100)')
df_all.columns = df_all.columns.str.strip()

# Filter to image-compatible prompts
df_img = df_all[df_all['Modality'].isin(['Image', 'Both'])].copy().reset_index(drop=True)

print(f'Total prompts    : {len(df_all)}')
print(f'Image-compatible : {len(df_img)}')
print(df_img.groupby(['Type','User Profile']).size().to_string())

# ── Load model ───────────────────────────────────────────────────────────────
import torch
import gc
from diffusers import StableDiffusion3Pipeline
from huggingface_hub import login

if HF_TOKEN:
    login(HF_TOKEN)

pipe = StableDiffusion3Pipeline.from_pretrained(
    HF_MODEL_ID,
    torch_dtype=torch.float16,
    token=HF_TOKEN,
)
pipe.enable_model_cpu_offload()  # SD3.5 (esp. Large) exceeds 8GB VRAM on this card; keeps it within budget
pipe.vae.enable_tiling()
pipe.set_progress_bar_config(disable=True)
print(f'{MODEL_LABEL} loaded on {pipe.device}')

# ── Generation helpers ───────────────────────────────────────────────────────
import hashlib, datetime

def make_filename(prompt_id, prompt_type, index=0):
    ts = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
    h  = hashlib.sha256(f'{prompt_id}_{index}_{ts}'.encode()).hexdigest()[:10]
    t  = 'amb' if prompt_type == 'Ambiguous' else 'exp'
    return f'P{prompt_id:03d}_{t}_{h}.png'

print('Helpers ready.')

# ── Resume: load already-generated index ─────────────────────────────────────
if os.path.exists(INDEX_CSV):
    df_done  = pd.read_csv(INDEX_CSV)
    done_ids = set(zip(df_done['prompt_id'], df_done['img_index']))
    print(f'Resuming: {len(df_done)} images already generated.')
else:
    df_done  = pd.DataFrame()
    done_ids = set()
    print('Starting fresh.')

# ── Main generation loop ─────────────────────────────────────────────────────
from tqdm.auto import tqdm

# SD 3.5 turbo uses cfg=0
_guidance = 0.0 if MODEL_KEY == 'turbo' else GUIDANCE_SCALE
_steps    = 4   if MODEL_KEY == 'turbo' else NUM_INFERENCE_STEPS

records = []

for _, row in tqdm(df_img.iterrows(), total=len(df_img), desc=MODEL_LABEL):
    pid         = int(row['ID'])
    ptype       = row['Type']
    prompt_text = str(row['Prompt Text'])

    for img_idx in range(IMAGES_PER_PROMPT):
        if (pid, img_idx) in done_ids:
            continue

        status = 'ok'
        fname  = ''
        try:
            with torch.no_grad():
                result = pipe(
                    prompt               = prompt_text,
                    num_inference_steps  = _steps,
                    guidance_scale       = _guidance,
                    num_images_per_prompt = 1,
                )
            image = result.images[0]
            fname = make_filename(pid, ptype, img_idx)
            image.save(os.path.join(OUTPUT_DIR, fname))
            done_ids.add((pid, img_idx))
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f'  Error P{pid:03d}: {str(e)[:100]}')
            status = 'error'

        records.append({
            'file_path'   : f'images/{MODEL_LABEL}/{fname}' if fname else '',
            'prompt_id'   : pid,
            'img_index'   : img_idx,
            'prompt_type' : ptype,
            'user_profile': row['User Profile'],
            'modality'    : row['Modality'],
            'category'    : row['Category'],
            'language'    : row['Language'],
            'status'      : status,
            'model'       : MODEL_LABEL,
            'prompt_text' : prompt_text,
        })

df_new = pd.DataFrame(records)
df_out = pd.concat([df_done, df_new], ignore_index=True) if not df_done.empty else df_new
df_out.to_csv(INDEX_CSV, index=False)

ok  = (df_out['status'] == 'ok').sum()
err = (df_out['status'] == 'error').sum()
print(f'\nDone. Generated: {ok} | Errors: {err}')
print(f'Index: {INDEX_CSV}')

# ── Summary ──────────────────────────────────────────────────────────────────
df_results = pd.read_csv(INDEX_CSV)
print(f'=== {MODEL_LABEL} Generation Summary ===')
print(df_results.groupby(['prompt_type','status']).size().unstack(fill_value=0).to_string())
print(f"\nTotal images saved: {(df_results['status']=='ok').sum()} / {len(df_results)}")
