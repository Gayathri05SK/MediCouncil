# MediCouncil — Railway Deployment Guide

> Follow this guide step by step to deploy MediCouncil on Railway.

---

## Prerequisites

- A [Railway](https://railway.app) account (sign up with GitHub)
- Access to the GitHub repo: `https://github.com/Gayathri05SK/MediCouncil`
- The three API keys:
  - `DEEPSEEK_API_KEY` — from [platform.deepseek.com](https://platform.deepseek.com)
  - `HF_API_KEY` — from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
  - `GLM_API_KEY` — from [open.bigmodel.cn](https://open.bigmodel.cn)

---

## Step 1 — Push trained models to GitHub (done by Bhav)

Before deploying, make sure the repo has the trained model files committed.
The following files must exist in the repo:

```
models/ml_baselines/naive_bayes.pkl
models/ml_baselines/logistic_regression.pkl
models/ml_baselines/random_forest.pkl
models/feature_builder.pkl
data/raw/disease_symptoms.csv
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
```

---

## Step 2 — Deploy on Railway

### 2.1 Create a new project

1. Go to [railway.app](https://railway.app) and log in
2. Click **New Project**
3. Select **Deploy from GitHub repo**
4. Search for and select `Gayathri05SK/MediCouncil`
5. Click **Deploy Now**

Railway will automatically detect Python and start building.

---

### 2.2 Set environment variables

Once the project is created, go to your service → **Variables** tab → **Add Variable** and add each of the following:

| Variable | Value |
|---|---|
| `DEEPSEEK_API_KEY` | your DeepSeek API key |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` |
| `HF_API_KEY` | your HuggingFace API token |
| `GPT_OSS_BASE_URL` | `https://api-inference.huggingface.co` |
| `GLM_API_KEY` | your ZhipuAI API key |
| `GLM_BASE_URL` | `https://open.bigmodel.cn/api/paas/v4` |
| `EMERGENCY_WEIGHT` | `0.5` |
| `GUIDELINE_WEIGHT` | `0.3` |
| `PRIMARY_WEIGHT` | `0.2` |
| `SAFETY_OVERRIDE_THRESHOLD` | `0.75` |
| `LOW_CONFIDENCE_THRESHOLD` | `0.6` |
| `API_HOST` | `0.0.0.0` |
| `DEBUG` | `False` |
| `LOG_LEVEL` | `INFO` |
| `LOG_FILE` | `medicouncil.log` |

> `PORT` is set automatically by Railway — do NOT add it manually.

After adding all variables, Railway will automatically redeploy.

---

### 2.3 Verify the deployment

1. Go to the **Deployments** tab and wait for the build to show **Active**
2. Click **View Logs** to check for any errors
3. Go to the **Settings** tab → **Networking** → click **Generate Domain**
4. Open the generated URL (e.g. `https://medicouncil-production.up.railway.app`)

You should see the MediCouncil frontend.

---

## Step 3 — Test the API

Visit `https://your-railway-url.up.railway.app/docs` to access the interactive Swagger UI and test the `/api/triage` endpoint directly in the browser.

**Sample test input:**
```json
{
  "symptoms_text": "severe chest pain radiating to left arm with sweating",
  "age": 55,
  "sex": "M",
  "chronic_conditions": ["hypertension", "diabetes"],
  "red_flags": ["chest_pain", "difficulty_breathing"]
}
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Build fails with `ModuleNotFoundError` | Make sure `requirements.txt` is in the repo root |
| App crashes on startup | Check that `.pkl` model files are committed to the repo |
| `PORT` error | Do NOT set `PORT` manually — Railway injects it automatically |
| LLM agents all fail | Double-check your API keys in the Variables tab |
| 500 error on `/api/triage` | Check deployment logs for the exact error |

---

## Local Setup (for reference)

If you want to run locally before deploying:

```bash
git clone https://github.com/Gayathri05SK/MediCouncil.git
cd MediCouncil
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
python scripts/prepare_data.py
python scripts/train_ml_models.py
python api/main.py
```

Then open `http://localhost:8000`.
