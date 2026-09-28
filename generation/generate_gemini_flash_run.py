# ── Configuration (edit these) ───────────────────────────────────────────────
import os
from dotenv import load_dotenv
load_dotenv()

GCP_PROJECT  = os.environ.get('GCP_PROJECT', 'secret-antonym-494814-q9')
GCP_LOCATION = os.environ.get('GCP_LOCATION', 'us-central1')
MODEL_NAME   = 'gemini-2.5-flash-image'   # image-generation model
IMAGES_PER_PROMPT = 1                      # set to 2-3 for more samples per prompt
MAX_API_CALLS     = 30                     # max retries per prompt before giving up

# Paths (relative to repo root — adjust if running from elsewhere)
REPO_ROOT   = os.path.abspath(os.path.join(os.getcwd(), '..'))
PROMPTS_XLS = os.path.join(REPO_ROOT, 'prompts', 'all_prompts.xlsx')
OUTPUT_DIR  = os.path.join(REPO_ROOT, 'dataset', 'media', 'images', 'Gemini_Flash')
INDEX_CSV   = os.path.join(REPO_ROOT, 'dataset', 'labels', 'generated_Gemini_Flash.csv')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(INDEX_CSV), exist_ok=True)

print(f'GCP Project : {GCP_PROJECT}')
print(f'Output dir  : {OUTPUT_DIR}')
print(f'Prompts xlsx: {PROMPTS_XLS}')

# ── Load prompts ─────────────────────────────────────────────────────────────
import pandas as pd

df_all = pd.read_excel(PROMPTS_XLS, sheet_name='All Prompts (100)')
df_all.columns = df_all.columns.str.strip()

# Filter to prompts that include images (Image or Both)
df_img = df_all[df_all['Modality'].isin(['Image', 'Both'])].copy().reset_index(drop=True)

print(f'Total prompts   : {len(df_all)}')
print(f'Image-compatible: {len(df_img)}')
print(f"  Ambiguous : {(df_img['Type']=='Ambiguous').sum()}")
print(f"  Explicit  : {(df_img['Type']=='Explicit').sum()}")
df_img[['ID', 'Type', 'User Profile', 'Modality', 'Category']].head(5)

# ── SDK init ─────────────────────────────────────────────────────────────────
import time, base64, hashlib, datetime
from google import genai
from google.genai import types

REGIONS = ['us-central1', 'europe-west1', 'asia-northeast1']
if GCP_LOCATION in REGIONS:
    REGIONS.remove(GCP_LOCATION)
    REGIONS.insert(0, GCP_LOCATION)

client = genai.Client(vertexai=True, project=GCP_PROJECT, location=REGIONS[0])
print(f'Vertex AI client initialised ({REGIONS[0]})')

# ── Generation helpers ───────────────────────────────────────────────────────
def make_filename(prompt_id, prompt_type, index=0):
    ts = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
    h  = hashlib.sha256(f'{prompt_id}_{index}_{ts}'.encode()).hexdigest()[:10]
    t  = 'amb' if prompt_type == 'Ambiguous' else 'exp'
    return f'P{prompt_id:03d}_{t}_{h}.png'


SAFETY_OFF = [
    types.SafetySetting(category='HARM_CATEGORY_HATE_SPEECH',       threshold='OFF'),
    types.SafetySetting(category='HARM_CATEGORY_DANGEROUS_CONTENT', threshold='OFF'),
    types.SafetySetting(category='HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold='OFF'),
    types.SafetySetting(category='HARM_CATEGORY_HARASSMENT',        threshold='OFF'),
]


def generate_image(prompt_text, region_idx=0):
    """Returns (image_bytes, status) where status in {ok, blocked, rate_limit, error}."""
    global client
    contents = [types.Content(role='user', parts=[types.Part.from_text(text=prompt_text)])]
    config   = types.GenerateContentConfig(
        temperature=1, top_p=0.95,
        response_modalities=['IMAGE'],
        safety_settings=SAFETY_OFF,
    )
    for attempt in range(4):
        try:
            resp = client.models.generate_content(
                model=MODEL_NAME, contents=contents, config=config
            )
            if not resp.candidates:
                return None, 'blocked'
            cand = resp.candidates[0]
            if hasattr(cand, 'finish_reason') and cand.finish_reason and cand.finish_reason.name == 'SAFETY':
                return None, 'blocked'
            if not cand.content or not cand.content.parts:
                return None, 'blocked'
            for part in cand.content.parts:
                if part.inline_data and part.inline_data.mime_type.startswith('image/'):
                    raw = part.inline_data.data
                    return (base64.b64decode(raw) if isinstance(raw, str) else raw), 'ok'
            return None, 'blocked'
        except Exception as e:
            err = str(e)
            if '429' in err or 'RESOURCE_EXHAUSTED' in err:
                wait = 10 * (2 ** attempt)
                print(f'  429 rate-limit — wait {wait}s (attempt {attempt+1}/4)')
                time.sleep(wait)
            else:
                print(f'  error: {err[:120]}')
                return None, 'error'
    return None, 'rate_limit'


print('Helpers ready.')

# ── Resume: load already-generated index ─────────────────────────────────────
if os.path.exists(INDEX_CSV):
    df_done = pd.read_csv(INDEX_CSV)
    done_ids = set(zip(df_done['prompt_type'], df_done['prompt_id'], df_done['img_index']))
    print(f'Resuming: {len(df_done)} images already generated.')
else:
    df_done  = pd.DataFrame()
    done_ids = set()
    print('Starting fresh.')

# ── Main generation loop ─────────────────────────────────────────────────────
from tqdm import tqdm
import json

records = []
region_idx = 0

for _, row in tqdm(df_img.iterrows(), total=len(df_img), desc='Generating'):
    pid  = int(row['ID'])
    ptype = row['Type']
    prompt_text = str(row['Prompt Text'])

    for img_idx in range(IMAGES_PER_PROMPT):
        if (ptype, pid, img_idx) in done_ids:
            continue   # already generated in a previous run

        img_bytes, status = generate_image(prompt_text, region_idx)

        if status == 'rate_limit':
            # Try next region
            region_idx = (region_idx + 1) % len(REGIONS)
            new_region = REGIONS[region_idx]
            print(f'  Switching region -> {new_region}')
            client = genai.Client(vertexai=True, project=GCP_PROJECT, location=new_region)
            time.sleep(5)
            img_bytes, status = generate_image(prompt_text, region_idx)

        record = {
            'file_path'   : '',
            'prompt_id'   : pid,
            'img_index'   : img_idx,
            'prompt_type' : ptype,
            'user_profile': row['User Profile'],
            'modality'    : row['Modality'],
            'category'    : row['Category'],
            'language'    : row['Language'],
            'status'      : status,
            'model'       : 'Gemini_Flash',
            'prompt_text' : prompt_text,
        }

        if status == 'ok' and img_bytes:
            fname = make_filename(pid, ptype, img_idx)
            fpath = os.path.join(OUTPUT_DIR, fname)
            with open(fpath, 'wb') as f:
                f.write(img_bytes)
            record['file_path'] = f'images/Gemini_Flash/{fname}'
            done_ids.add((ptype, pid, img_idx))

        records.append(record)
        time.sleep(1.5)  # avoid burst throttling

# Merge with previous run and save
df_new  = pd.DataFrame(records)
df_all_results = pd.concat([df_done, df_new], ignore_index=True) if not df_done.empty else df_new
df_all_results.to_csv(INDEX_CSV, index=False)

ok    = (df_all_results['status'] == 'ok').sum()
blk   = (df_all_results['status'] == 'blocked').sum()
err   = df_all_results['status'].isin(['error','rate_limit']).sum()
print(f'\nDone. Generated: {ok} | Blocked: {blk} | Error/Limit: {err}')
print(f'Index saved: {INDEX_CSV}')

# ── Summary ──────────────────────────────────────────────────────────────────
df_results = pd.read_csv(INDEX_CSV)
print('=== Generation Summary ===')
print(df_results.groupby(['prompt_type','status']).size().unstack(fill_value=0).to_string())
print(f"\nTotal images saved: {(df_results['status']=='ok').sum()}")
print(f"Blocked by safety: {(df_results['status']=='blocked').sum()}")
print(f"\nBlocked prompt examples:")
blocked = df_results[df_results['status']=='blocked'][['prompt_id','prompt_type','category','prompt_text']]
if not blocked.empty:
    for _, r in blocked.head(5).iterrows():
        print(f"  P{r['prompt_id']:03d} [{r['prompt_type']}] {r['category'][:50]}")
