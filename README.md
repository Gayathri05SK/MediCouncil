# MediCouncil: Multi-Agent LLM Council for Symptom Triage

> An AI-powered clinical decision support system that uses a council of specialized Large Language Models to perform safe, explainable symptom triage.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [LLM Council Agents](#llm-council-agents)
- [ML Baselines](#ml-baselines)
- [Consensus Engine](#consensus-engine)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [API Reference](#api-reference)
- [Scripts](#scripts)
- [Tech Stack](#tech-stack)
- [Disclaimer](#disclaimer)

---

## Overview

MediCouncil is a multi-agent clinical decision support system built for symptom triage. It combines three specialized LLM agents that independently assess patient symptoms and reach a weighted consensus decision — prioritizing patient safety above all else.

The system is **not a replacement for medical professionals**. It is a decision support tool that flags urgency levels, estimates risk scores, and provides reasoned explanations to assist triage.

### Key Features

- **Multi-Agent Council**: Three LLMs (Gemini,Groq models) run in parallel, each specializing in a different clinical perspective
- **Safety-First Override**: If any agent detects an emergency condition with high confidence, it immediately overrides the consensus — critical cases are never downgraded
- **Weighted Consensus**: Agents are weighted by role (Emergency 50%, Guideline 30%, Primary Care 20%) and their outputs are merged into a final risk score and urgency level
- **ML Baselines**: Naive Bayes, Logistic Regression, and Random Forest classifiers run alongside the LLMs for benchmarking
- **Explainable Output**: Each agent provides its reasoning; the consensus engine synthesizes these into a human-readable explanation
- **REST API**: FastAPI backend with Swagger docs at `/docs`
- **Web Frontend**: Simple HTML/CSS/JS interface for symptom input and result visualization

---

## Architecture

```
Patient Input (Symptoms, Age, Sex, Chronic Conditions)
            │
            ▼
    ┌───────────────┐
    │FeatureBuilder │  ← Preprocesses symptoms into ML features
    └───────┬───────┘
            │
     ┌──────┴──────┐
     │             │
     ▼             ▼
┌─────────┐  ┌──────────────────────────────────────┐
│ML Models│  │           LLM Council                 │
│  NB     │  │  ┌────────────┐  ┌────────────────┐  │
│  LR     │  │  │Emergency   │  │Primary Care    │  │
│  RF     │  │  │LLama3.3-Groq│ │Gemini-flash    │  │
└────┬────┘  │  └────────────┘  └────────────────┘  │
     │       │       ┌────────────────┐              │
     │       │       │Guideline Agent │              │
     │       │       │LLama3.1-Groq   │              │
     │       │       └────────────────┘              │
     │       └──────────────────┬───────────────────┘
     │                          │
     └──────────┬───────────────┘
                ▼
      ┌──────────────────┐
      │ Consensus Engine  │
      │ - Safety Override │
      │ - Weighted Vote   │
      │ - Confidence Score│
      └────────┬─────────┘
               ▼
     Final Triage Decision
  (Risk Level, Urgency, Explanation)
```

---

## LLM Council Agents

Each agent runs asynchronously and independently evaluates the patient case from its specialized role.

| Agent | Model | Role | Weight | Temperature |
|---|---|---|---|---|
| Emergency | LLama3.3 | Detects critical/life-threatening conditions | 50% | 0.2 |
| Primary Care | Gemini-flash | General symptom assessment | 20% | 0.4 |
| Guideline | LLama3.1 | Evidence-based clinical guidelines | 30% | 0.3 |

Each agent returns:
- `risk_score` (0–100)
- `urgency` (`emergency` / `urgent` / `routine` / `self_care`)
- `confidence` (0.0–1.0)
- `reasoning` (natural language explanation)
- `key_symptoms` (list of flagged symptoms)

---

## ML Baselines

Three classical ML models are trained on the symptom dataset and run in parallel with the LLM council. They serve as benchmarks to compare against the LLM-based approach.

| Model | Algorithm |
|---|---|
| Naive Bayes | `MultinomialNB` |
| Logistic Regression | `LogisticRegression` (balanced, max_iter=1000) |
| Random Forest | `RandomForestClassifier` (200 trees, balanced) |

---

## Consensus Engine

The `ConsensusEngine` merges outputs from all three agents using the following logic:

1. **Safety Override Check**: If any agent flags `urgency = emergency` with `risk_score >= 75` or `confidence >= 0.75`, the final decision is immediately set to emergency — no voting occurs.
2. **Weighted Risk Score**: Each agent's risk score is multiplied by its weight and averaged.
3. **Weighted Urgency**: Urgency levels are mapped to integers and averaged with weights.
4. **Confidence**: Computed from inter-agent agreement (60%) and average agent confidence (40%).
5. **Explanation**: Each agent's reasoning is combined into a final explanation with a clinical recommendation.

**Urgency Levels:**
| Level | Recommendation |
|---|---|
| `emergency` | Seek immediate emergency care |
| `urgent` | See a provider within 24 hours |
| `routine` | Schedule a routine appointment |
| `self_care` | Monitor symptoms; self-care may be sufficient |

---

## Project Structure

```
medicouncil/
├── api/
│   ├── main.py              # FastAPI app entry point
│   ├── routes.py            # API route definitions
│   ├── schemas.py           # Pydantic request/response models
│   └── __init__.py
├── config/
│   ├── settings.py          # All configuration (API keys, model params, paths)
│   ├── prompts.py           # System prompts for each LLM agent
│   └── __init__.py
├── data/
│   ├── raw/                 # Raw dataset (disease_symptoms.csv)
│   ├── processed/           # Processed features and labels
│   └── synthetic/           # Synthetic patient cases for testing
├── frontend/
│   ├── static/
│   │   ├── css/style.css
│   │   └── js/app.js
│   └── templates/
│       ├── index.html       # Symptom input form
│       └── results.html     # Triage result display
├── models/
│   └── ml_baselines/        # Saved .pkl files for trained ML models
├── results/
│   ├── metrics/             # Evaluation metrics (JSON/CSV)
│   ├── predictions/         # Prediction outputs
│   └── reports/             # Generated evaluation reports
├── scripts/
│   ├── prepare_data.py      # Data preprocessing pipeline
│   ├── train_ml_models.py   # Train and save ML baseline models
│   ├── evaluate_system.py   # Full system evaluation
│   └── test_inference.py    # Quick inference test
├── src/
│   ├── data/
│   │   ├── dataset_builder.py
│   │   ├── feature_builder.py
│   │   └── preprocessor.py
│   ├── models/
│   │   ├── llm_council.py       # LLM agent orchestration
│   │   ├── consensus_engine.py  # Weighted voting and safety logic
│   │   └── ml_baselines.py      # ML model wrappers
│   ├── inference/
│   │   └── pipeline.py          # End-to-end inference pipeline
│   └── evaluation/
│       ├── metrics.py
│       └── visualizer.py
├── logs/                    # Runtime logs
├── requirements.txt
├── .env                     # API keys (not committed)
└── .gitignore
```

---

## Setup & Installation

### Prerequisites

- Python 3.9 or higher
- API keys for Gemini, and Groq

### 1. Clone the repository

```bash
git clone https://github.com/MediCouncil.git
cd MediCouncil
```

### 2. Create a virtual environment

```bash
python -m venv medicouncil_env
source medicouncil_env/bin/activate      # Linux/macOS
medicouncil_env\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
# LLM API Keys
GROQ_API_KEY=your_api_key
GOOGLE_API_KEY=your_api_key



# Consensus Weights (must sum to 1.0)
EMERGENCY_WEIGHT=0.5
GUIDELINE_WEIGHT=0.3
PRIMARY_WEIGHT=0.2

SAFETY_OVERRIDE_THRESHOLD=0.75
LOW_CONFIDENCE_THRESHOLD=0.6
# Logging
LOG_LEVEL=INFO
LOG_DIR=logs
LOG_FILE=medicouncil.log

# Consensus Engine
CONSENSUS_ALGORITHM=weighted
MIN_CONFIDENCE_THRESHOLD=0.6
MAX_DELIBERATION_ROUNDS=3

DATABASE_URL=sqlite:///./medicouncil.db
# Server
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

```

---

## Running the Application

### Step 1: Prepare the data

```bash
python scripts/prepare_data.py
```

### Step 2: Train ML baseline models

```bash
python scripts/train_ml_models.py
```

### Step 3: Start the API server

```bash
python api/main.py
```

The server starts at `http://localhost:8000`.

- **Frontend**: `http://localhost:8000/`
- **Swagger API Docs**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## API Reference

### `POST /api/triage`

Submit a patient case for triage.

**Request Body:**
```json
{
  "symptoms_text": "severe chest pain radiating to left arm, shortness of breath",
  "age": 55,
  "sex": "male",
  "chronic_conditions": ["hypertension", "diabetes"],
  "red_flags": ["chest pain", "shortness of breath"]
}
```

**Response:**
```json
{
  "risk_level": "Critical",
  "risk_score": 88,
  "urgency": "emergency",
  "confidence": 0.91,
  "confidence_level": "High",
  "explanation": "...",
  "key_symptoms": ["chest pain", "shortness of breath"],
  "agent_agreement": { "rate": 1.0, "level": "high" },
  "safety_override": true,
  "ml_baseline": { ... },
  "disclaimer": "..."
}
```

### `GET /api/health`

Returns server health status.

---

## Scripts

| Script | Purpose |
|---|---|
| `scripts/prepare_data.py` | Load raw CSV, preprocess, build features |
| `scripts/train_ml_models.py` | Train Naive Bayes, LR, Random Forest and save models |
| `scripts/evaluate_system.py` | Run full evaluation on test cases and generate reports |
| `scripts/test_inference.py` | Quick single-case inference test |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI, Uvicorn |
| LLM Agents | Llama,Gemini |
| ML Baselines | scikit-learn (Naive Bayes, Logistic Regression, Random Forest) |
| Async HTTP | aiohttp |
| Data | pandas, numpy |
| Visualization | matplotlib, seaborn, plotly |
| Frontend | HTML, CSS, JavaScript (Jinja2 templates) |
| Config | python-dotenv, pyyaml |
| Testing | pytest, pytest-asyncio, httpx |

---

## Disclaimer

**MediCouncil is a research and decision support tool — it is NOT a substitute for professional medical advice, diagnosis, or treatment.**

In case of a medical emergency, call your local emergency number immediately (e.g., **112** / **108** / **911**).

Always consult a qualified healthcare provider for proper diagnosis and treatment decisions.
