"""AIME — Gemini 2.5 Flash: generate UNSAFE images from dataset_prompts_classified_bilingual.json."""
import os, sys, time, base64, hashlib, datetime
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
GCP_PROJECT  = os.environ.get('GCP_PROJECT', 'secret-antonym-494814-q9')
MODEL_NAME   = 'gemini-2.5-flash-image'
IMAGES_PER_PROMPT = 1

REPO_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_JSON    = os.path.join(REPO_ROOT, 'prompts', 'dataset_prompts_classified_bilingual.json')
OUTPUT_DIR  = os.path.join(REPO_ROOT, 'dataset', 'media', 'images', 'Gemini_Flash_Unsafe')
INDEX_CSV   = os.path.join(REPO_ROOT, 'dataset', 'labels', 'generated_Gemini_Flash_unsafe.csv')
BLOCKED_CSV = os.path.join(REPO_ROOT, 'dataset', 'labels', 'generated_Gemini_Flash_unsafe_blocked.csv')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(INDEX_CSV), exist_ok=True)

print(f'Source JSON : {SRC_JSON}')
print(f'Output dir  : {OUTPUT_DIR}')

# ── Load prompts ──────────────────────────────────────────────────────────────
import json, pandas as pd

with open(SRC_JSON, encoding='utf-8') as f:
    data = json.load(f)

print(f'Loaded {len(data)} prompt pairs — using unsafe_prompt_en field')

# ── SDK init ──────────────────────────────────────────────────────────────────
from google import genai
from google.genai import types

REGIONS = ['us-central1', 'europe-west1', 'asia-northeast1']
client  = genai.Client(vertexai=True, project=GCP_PROJECT, location=REGIONS[0])
print(f'Vertex AI client ready ({REGIONS[0]})\n')

# ── Helpers ───────────────────────────────────────────────────────────────────
def make_filename(global_id, index=0):
    ts = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
    h  = hashlib.sha256(f'unsafe_{global_id}_{index}_{ts}'.encode()).hexdigest()[:10]
    return f'U{global_id:04d}_unsafe_{h}.png'

SAFETY_OFF = [
    types.SafetySetting(category='HARM_CATEGORY_HATE_SPEECH',       threshold='OFF'),
    types.SafetySetting(category='HARM_CATEGORY_DANGEROUS_CONTENT', threshold='OFF'),
    types.SafetySetting(category='HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold='OFF'),
    types.SafetySetting(category='HARM_CATEGORY_HARASSMENT',        threshold='OFF'),
]

def generate_image(prompt_text, region_idx=0):
    global client
    contents = [types.Content(role='user', parts=[types.Part.from_text(text=prompt_text)])]
    config   = types.GenerateContentConfig(
        temperature=1, top_p=0.95,
        response_modalities=['IMAGE'],
        safety_settings=SAFETY_OFF,
    )
    for attempt in range(4):
        try:
            resp = client.models.generate_content(model=MODEL_NAME, contents=contents, config=config)
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
                print(f'  429 rate-limit — wait {wait}s (attempt {attempt+1}/4)', flush=True)
                time.sleep(wait)
            else:
                print(f'  error: {err[:120]}', flush=True)
                return None, 'error'
    return None, 'rate_limit'

# ── Resume ────────────────────────────────────────────────────────────────────
if os.path.exists(INDEX_CSV):
    df_done  = pd.read_csv(INDEX_CSV)
    done_ids = set(df_done['global_id'].tolist())
    print(f'Resuming: {len(df_done)} images already generated.')
else:
    df_done  = pd.DataFrame()
    done_ids = set()
    print('Starting fresh.')

# ── Main loop ─────────────────────────────────────────────────────────────────
records    = []
region_idx = 0
total      = len(data)

for i, item in enumerate(data):
    gid         = item['global_id']
    prompt_text = item['unsafe_prompt_en']   # ← UNSAFE prompt in English

    if gid in done_ids:
        print(f'[{i+1:4d}/{total}] G{gid:04d} skip (already done)')
        continue

    print(f'[{i+1:4d}/{total}] G{gid:04d} [{item["category"][:20]:<20}] generating...', end=' ', flush=True)
    img_bytes, status = generate_image(prompt_text, region_idx)

    if status == 'rate_limit':
        region_idx = (region_idx + 1) % len(REGIONS)
        new_region = REGIONS[region_idx]
        print(f'rate-limit → switching to {new_region}', flush=True)
        client = genai.Client(vertexai=True, project=GCP_PROJECT, location=new_region)
        time.sleep(5)
        img_bytes, status = generate_image(prompt_text, region_idx)

    record = {
        'file_path'       : '',
        'global_id'       : gid,
        'category_id'     : item['id'],
        'category'        : item['category'],
        'type'            : item['type'],
        'user_profile'    : item['user_profile'],
        'modality'        : item['modality'],
        'label'           : 'unsafe',
        'status'          : status,
        'model'           : 'Gemini_Flash',
        'prompt_text_en'  : prompt_text,
        'prompt_text_it'  : item['unsafe_prompt_it'],
    }

    if status == 'ok' and img_bytes:
        fname = make_filename(gid)
        fpath = os.path.join(OUTPUT_DIR, fname)
        with open(fpath, 'wb') as f:
            f.write(img_bytes)
        record['file_path'] = f'images/Gemini_Flash_Unsafe/{fname}'
        done_ids.add(gid)
        print(f'OK  ({round(len(img_bytes)/1024)}KB) → {fname}', flush=True)
    else:
        print(f'{status.upper()}', flush=True)

    records.append(record)

    if len(records) % 10 == 0:
        df_new = pd.DataFrame(records)
        df_out = pd.concat([df_done, df_new], ignore_index=True) if not df_done.empty else df_new
        df_out.to_csv(INDEX_CSV, index=False)

    time.sleep(1.0)

# ── Final save + blocked export ───────────────────────────────────────────────
df_new = pd.DataFrame(records)
df_out = pd.concat([df_done, df_new], ignore_index=True) if not df_done.empty else df_new
df_out.to_csv(INDEX_CSV, index=False)

blocked = df_out[df_out['status'].isin(['blocked', 'error', 'rate_limit'])]
if not blocked.empty:
    blocked.to_csv(BLOCKED_CSV, index=False)

ok  = (df_out['status'] == 'ok').sum()
blk = (df_out['status'] == 'blocked').sum()
err = df_out['status'].isin(['error', 'rate_limit']).sum()

print(f'\n{"="*55}')
print(f'DONE — Generated: {ok} | Blocked: {blk} | Error/Limit: {err}')
print(f'Index  : {INDEX_CSV}')
if not blocked.empty:
    print(f'Blocked: {BLOCKED_CSV}')
