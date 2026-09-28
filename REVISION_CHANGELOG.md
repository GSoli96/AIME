# AIME Paper — Revision Changelog

Tracking document for the major revision of:
> *"Exploring the Risks of Generative AI Misuse by Inexperienced Users"*
> Image and Vision Computing, Elsevier — Major Revision (2026)

---

## STATO ORIGINALE (pre-revisione)

### Prompt
| File | Contenuto |
|------|-----------|
| `prompts/ambiguous.txt` | 22 prompt ambigui |
| `prompts/explicit.txt` | 4 prompt espliciti |
| **Totale** | **26 prompt** |

- Nessuna distinzione formale tra utente inesperto ed esperto
- Solo lingua italiana/inglese mista, senza tag sistematici
- Categorie implicite: odio razziale, stereotipi, violenza, contenuto sessuale

### Dataset media
| Tipo | Quantità | Modelli |
|------|----------|---------|
| Immagini | **1.445** | Copilot (92), Playground v2.5 (23), SD 3.5 Large (450), SD 3.5 Medium (375), SD 3.5 Turbo (400), SD 3-2B (46), SDXL (50), Kaiber (9) |
| Video | **110** (64 non-etici) | Runway (38), Pika (26), Decohere (22), Genmo (14), Kaiber (4), Zeroscope (2), Pixverse (4) |
| **Totale** | **1.555 file** | **14 modelli (8 T2I + 6 T2V)** |

### Label
| File | Righe | Tipo |
|------|-------|------|
| `aime_binary_en.csv` | 1.555 | Binario: 0=etico, 1=non-etico |
| `aime_binary_it.csv` | 1.555 | Binario (italiano) |
| `aime_multiclass_en.csv` | 324 | Multiclasse: offensive / stereotypes / prejudices / violence / nothing |
| `aime_multiclass_it.csv` | 324 | Multiclasse (italiano) |

### Annotazione
- 3 esperti umani (domain experts)
- Metriche: accuracy, precision, recall, F1, Cohen's kappa (inter-annotator agreement)
- LLM evaluation: Gemini 1.5 Pro, Claude Haiku/Sonnet, GPT-4o

---

## AGGIUNTE PER LA REVISIONE

### R1C3 / R2C6 — Operazionalizzazione "utente inesperto"

#### Nuova tassonomia dei prompt
Aggiunto header sistematico a `ambiguous.txt` e `explicit.txt`:
- `[INEXPERIENCED USER]` — utente che genera contenuto dannoso accidentalmente, senza consapevolezza
- `[EXPERIENCED USER]` — utente che costruisce deliberatamente prompt per eludere i filtri
- Tag modalità: `[IMAGE]`, `[VIDEO]`, `[BOTH]`

#### Espansione prompt (26 → 100)
| File | Prima | Dopo | Delta |
|------|-------|------|-------|
| `prompts/ambiguous.txt` | 22 | **80** | +58 |
| `prompts/explicit.txt` | 4 | **20** | +16 |
| **Totale** | **26** | **100** | **+74** |

Nuove categorie coperte: cyberbullying, body shaming, stigma salute mentale, glorificazione droghe, discriminazione disabilità/età/classe, disturbi alimentari, anti-LGBTQ+, intolleranza religiosa, negazionismo Olocausto, incel/MGTOW, terrorismo stocastico, race science, poverty porn, anti-Rom, autolesionismo, white genocide conspiracy, matrimoni forzati, sorveglianza/voyeurismo.

**`prompts/all_prompts.xlsx`** — ricostruito ex-novo con 4 fogli:
- "All Prompts (100)": tutti i 100 prompt con colonne ID, Type, User Profile, Modality, Category, Language, Prompt Text
- "Ambiguous (80)", "Explicit (20)", "Statistics"
- Statistiche finali: 80 ambigui (23 Inexperienced + 57 Experienced), 20 espliciti; 43 Image-only, 19 Video-only, 38 Both; 52 IT + 48 EN

#### Nuovi prompt da utenti reali (dataset_prompts.jsonl)
Dataset raccolto da sessioni con utenti esperti e non esperti reali — **350 coppie** (safe + unsafe):

| Categoria | Coppie | Explicit | Ambiguous |
|-----------|--------|----------|-----------|
| Hate | 50 | 44 | 6 |
| Harassment | 50 | 37 | 13 |
| Violence | 50 | 40 | 10 |
| Self-Harm | 50 | 30 | 20 |
| Sexual | 50 | 33 | 17 |
| Shocking | 50 | 21 | 29 |
| Illegal Activity | 50 | 35 | 15 |
| **Totale** | **350** | **240** | **110** |

- Ogni prompt classificato con: Type, User Profile, Modality
- Traduzione in italiano via Gemini Flash
- File bilingue: `dataset_prompts_classified_bilingual.json`

---

### R1C1 — Scala del dataset ("64 video non-etici insufficienti")

#### Nuovo modello T2I: Gemini 2.5 Flash Image
- **Notebook**: `generation/generate_gemini_flash.ipynb`
- **Prompts**: 81 image-compatible (da all_prompts.xlsx, Modality ∈ {Image, Both})
- **Output**: `dataset/media/images/Gemini_Flash/`
- **Index CSV**: `dataset/labels/generated_Gemini_Flash.csv`
- Generazione in corso (luglio 2026)

#### Modelli SD 3.5 (già presenti, ora con nuovi prompt)
- **Notebook**: `generation/generate_SD35_aime.ipynb`
- Supporta: SD 3.5 Large / Medium / Turbo
- **Prompts**: stessi 81 image-compatible

#### Risultati generazione Gemini Flash (completata, tutti gli 81 prompt tentati — 2026-07-14)
| Fonte | Prompts | Generate | Bloccate | Errori |
|-------|---------|----------|----------|--------|
| Gemini 2.5 Flash | 81 | **36** | **45 (56%)** | 0 |

Nota: i blocchi sono concentrati sui prompt con encoding sofisticato (codici nazisti 88/1488, rune Odal SS, triple-parentesi antisemite, white genocide conspiracy) e su **tutti e 16 i prompt Explicit** (0/16 riusciti) — risultato di ricerca interessante di per sé (il filtro di sicurezza di Gemini è molto più severo sui prompt espliciti che su quelli ambigui).

**Bug corretto (2026-07-14)**: la logica di resume del notebook deduplicava solo su `prompt_id`, ignorando `prompt_type` — con ID numerici che si ripetono tra Ambiguous ed Explicit, questo faceva saltare per errore 8 prompt Explicit mai realmente tentati (venivano considerati "già fatti" perché un Ambiguous con lo stesso ID era stato processato prima). Corretto in `generate_gemini_flash.ipynb` (chiave ora `(prompt_type, prompt_id, img_index)`) e rieseguito: i tentativi ora coprono tutti gli 81/81, risultato: 0/8 nuovi successi (tutti bloccati dal safety filter, coerente con l'essere tutti Explicit).

Index CSV: `dataset/labels/generated_Gemini_Flash.csv`

#### Stima nuove immagini (da completare)
| Fonte | Prompts | Immagini previste |
|-------|---------|-------------------|
| Gemini 2.5 Flash | 81 | **36** (completato) |
| SD 3.5 Large | 81 | ~81 |
| SD 3.5 Medium | 81 | ~81 |
| SD 3.5 Turbo | 81 | ~81 |
| **Totale aggiuntivo** | | **~279** |

**Proiezione totale immagini**: 1.445 + ~279 = **~1.724 immagini**

---

### R1C5 — Aggiunta modelli T2V (Kling, Seedance2)

#### Modelli T2V da aggiungere
| Modello | Provider | Note |
|---------|----------|------|
| **Kling 3.0** | Kuaishou (klingai.com) | Richiesto esplicitamente dal revisore |
| **Seedance 2.5** | ByteDance (Doubao app) | Richiesto esplicitamente dal revisore |
| **Wan 2.1** | Wan-AI (HuggingFace: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, non 14B) | Open-source, Apache 2.0. Variante 1.3B scelta per starci in 8GB VRAM locali (14B non ci sta) |
| **HunyuanVideo** | Tencent (`tencent/HunyuanVideo`) | Open-source. Serve GPU cloud (~14GB anche quantizzato int4): non gira sulla GPU locale (8GB) |

- **Prompts video-compatible**: **57** esatti (19 Video-only + 38 Both, da `all_prompts.xlsx`)
- **Stima nuovi video**: 57 prompt × 4 modelli = **~228 video** aggiuntivi (proiezione aggiornata)
- **Proiezione totale video**: 110 + ~228 = **~338 video**

#### Stato notebook (creati 2026-07-02)
| Notebook | Stato | Note |
|---|---|---|
| `generation/generate_kling.ipynb` | Pronto, non testato | API verificata su docs ufficiali (JWT auth, `POST /v1/videos/text2video`). Serve `KLING_ACCESS_KEY`/`KLING_SECRET_KEY` in `.env` (kling.ai/dev/api-key) |
| `generation/generate_seedance.ipynb` | Pronto, non testato | Via BytePlus ModelArk. Serve `SEEDANCE_API_KEY` in `.env`. ⚠️ Model id di Seedance 2.5 non ancora confermato pubblicamente (GA prevista inizio luglio 2026) — il notebook usa di default l'id di Seedance 2.0, sovrascrivibile con `SEEDANCE_MODEL_ID` una volta confermato dalla console |
| `generation/generate_T2V_opensource.ipynb` | Wan2.1: pronto, smoke test non ancora eseguito (GPU occupata da SD3.5 al momento della creazione). HunyuanVideo: implementato ma quasi certamente OOM su GPU locale 8GB | Selettore `MODEL_KEY` come in `generate_SD35_aime.ipynb`; guard `SMOKE_TEST=True` incluso |

Env var aggiunte a `.env.example`; dipendenze aggiunte a `requirements.txt` (`ftfy`, `imageio`, `imageio-ffmpeg`, `bitsandbytes`, `requests`, `pyjwt`, `diffusers>=0.37`).

---

### Annotazione automatica — `test_soft_prompt_2` (completata, contribuisce a R1C1)

Fonte aggiuntiva di immagini candidate per la scala del dataset, oltre a Gemini Flash e SD 3.5:

- **4.505 immagini** JPG generate con prompt espliciti, modello **StableDiffusion3-2B** (identificato 2026-07-02)
- Categorie: harassment (692), hate (627), sexual (1.315), shocking (671), violence (471), self-harm (456), illegal-activity (270), unmatch (3)
- **Cambio di piano (deciso 2026-07-10)**: il piano originale di 3 annotatori umani (`Rev_1/ANNOTATION_GUIDE.md`) è stato **abbandonato, mai eseguito** — il set (4.712 immagini) era troppo grande per l'annotazione umana in tempi ragionevoli (risposta a R1C1, scala del dataset). Sostituito con 3 vision-language model locali/open-source come giudici indipendenti (binario + multiclasse, voto di maggioranza 2/3): **Qwen2.5-VL-3B-Instruct**, **LLaVA**, **Gemma3:4b**. Esclusa l'OpenAI Moderation API come giudice/alternativa: strutturalmente cieca su harassment/hate per immagini.
- **Risultati**: kappa a coppie basso su offensive/stereotypes/prejudices (0.04–0.36), solido su violence (fino a 0.76) — riportato onestamente come limite nel paper. Qwen2.5-VL-3B nettamente il più affidabile (86,5% accordo esatto multiclasse con l'oracolo, vs 54,5% LLaVA, 47,8% Gemma3). 157/4.712 immagini (3,3%) in triplo disaccordo, risolte con priorità violence > prejudices > offensive > stereotypes.
- **Deciso (2026-07-14)**: 3 annotatori restano sufficienti, nessun 4°/5° giudice via Ollama aggiunto.
- Dettagli completi: `TO_COPY_TO_MAIN_PROJECT/labels/RESULTS.md`
- **Stato**: CSV puliti (207 righe fantasma rimosse il 2026-07-10); immagini + label **non ancora copiate** in `AIME-public/dataset/` ufficiale

---

## RIEPILOGO DELTA TOTALE

| Risorsa | Prima | Dopo (previsto) | Delta |
|---------|-------|-----------------|-------|
| Prompt | 26 | 100 (+350 coppie da utenti) | +74 (+350) |
| Modelli T2I | 8 | 9 (+Gemini Flash) | +1 |
| Modelli T2V | 6 | 10 (+Kling, Seedance, Wan, Hunyuan) | +4 |
| Immagini | 1.445 | ~1.765 | ~+320 |
| Video | 110 | ~310 | ~+200 |
| **Media totali** | **1.555** | **~2.075** | **~+520** |

---

## TODO RIMANENTI

**Priorità: completare tutto ciò che riguarda le immagini prima di passare ai video (deciso 2026-07-14).**

- [x] Completare generazione immagini Gemini Flash — 36/81 generate (definitivo, tutti gli 81 tentati dopo il fix del bug di resume), 45 bloccate (vedi sopra)
- [~] Generare immagini con SD 3.5 Large/Medium/Turbo (nuovi 81 prompt) — **scoperto 2026-07-14: già eseguito su Colab il 2026-07-03**, output sincronizzato in `C:\Users\giand\Desktop\Annotazione\AIME\dataset\media\images\SD35_{Large,Medium,Turbo}\` (207 immagini reali, 69/81 prompt per modello — erano state erroneamente rimosse come "righe fantasma" il 2026-07-10 per ricerca incompleta, ora corretto). **Causa reale dei 12 prompt mancanti identificata (2026-07-14, non era un timeout di sessione come ipotizzato)**: bug nella resume-logic di `generate_SD35_colab.ipynb`, identico a quello trovato su `generate_gemini_flash.ipynb` — dedup key su solo `prompt_id` (non `prompt_type`), quindi i 12 prompt Explicit con lo stesso ID numerico di un Ambiguous già processato nello stesso run venivano saltati senza nemmeno essere tentati. Bug corretto in `generate_SD35_colab.ipynb`.
- [x] **Passaggio a GPU locale (2026-07-14)**: trovata una seconda macchina con RTX 4090 (24GB VRAM) — sufficiente per SD3.5 **full fp16, senza quantizzazione né offload** (il download su Colab era troppo lento). Creato `generate_SD35_local_4090.ipynb` con lo stesso fix del bug di resume, pronto per completare i 12 prompt mancanti × 3 modelli sulla nuova macchina. Nota: le 207 immagini già fatte su Colab sono quantizzate 4-bit (leggera incoerenza di qualità vs le nuove in fp16 pieno, considerata trascurabile per uso content-safety). Non ancora eseguito.
- [x] Copertura immagini complessiva: **69/81 prompt** hanno almeno un'immagine (Gemini o SD3.5); **12/81** ancora a zero immagini (gli stessi 12 Explicit sopra — bloccati anche su Gemini dove tentati)
- [x] Annotazione automatica `test_soft_prompt_2` (3 VLM, 4.712 immagini) — completata, CSV puliti (2026-07-10), metriche complete in RESULTS.md (2026-07-14). Decisione presa: 3 giudici bastano, nessun 4°/5° modello
- [ ] Copiare le 4.505 immagini + label pulite (`aime_binary_final_v2.csv`, `aime_multiclass_final_v2.csv`) da `TO_COPY_TO_MAIN_PROJECT/` in `AIME-public/dataset/` ufficiale
- [ ] Scrivere paragrafo methods/limitations sulla deviazione umano→automatico nell'annotazione
- [ ] Aggiornare label CSV con nuovi media (Gemini Flash + SD3.5 quando pronto)
- [ ] LLM evaluation nuove immagini (Claude, GPT-4o)
- [ ] Upload su HuggingFace `DAISLab-Unisa/AIME-Dataset`
- [ ] Aggiornare paper: sezione dataset, tabelle risultati, related work (T2VSafetyBench), limitations
- [ ] Rispondere ai singoli commenti dei revisori (rev_comment.tex)

### In stallo — video (deciso 2026-07-14: prima si chiudono tutti i task immagini)
- [~] Creare notebook T2V per Kling, Seedance, Wan2.1, HunyuanVideo — notebook creati (2026-07-02), nessuna generazione eseguita: Kling/Seedance bloccati su API keys mancanti, Wan2.1 in attesa di smoke test, HunyuanVideo richiede GPU cloud. **Non riprendere finché i task immagine sopra non sono chiusi.**
- [x] Aggiunti 2 modelli T2V gratuiti senza API key a `generate_T2V_opensource.ipynb` (2026-07-14): **CogVideoX-2B** (`zai-org/CogVideoX-2b`, Apache 2.0) e **LTX-Video-0.9.7-distilled** (`Lightricks/LTX-Video-0.9.7-distilled`, licenza uso ricerca) — entrambi girano su 8GB locali con offload. Notebook ora supporta 4 `MODEL_KEY`: `wan21`, `cogvideox`, `ltxvideo` (tutti gratis, no API key), `hunyuanvideo` (richiede GPU cloud). Ollama verificato: **non supporta modelli T2V** (architettura per LLM/VLM via llama.cpp, non pipeline di diffusione). Corretto preventivamente lo stesso bug di resume (dedup key mancante di `prompt_type`) trovato su `generate_gemini_flash.ipynb`. Nessuna generazione ancora eseguita — resta in stallo con gli altri T2V.
