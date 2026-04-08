# 🏥 MediCouncil: Multi-Agent LLM Council for Symptom Triage

## Overview

MediCouncil is an AI-powered decision support system that performs symptom triage using a heterogeneous council of Large Language Models (LLMs). The system combines classical machine learning baselines with specialized LLM agents to provide safe, explainable triage recommendations.

### Key Features

- **Multi-Agent LLM Council**: Three specialized agents (Emergency, Primary Care, Guideline) provide independent assessments
- **Safety-First Consensus**: Emergency override logic ensures critical cases are never missed
- **Classical ML Baselines**: Naive Bayes, Logistic Regression, Random Forest for benchmarking
- **Explainable Outputs**: Detailed reasoning from each agent with confidence scores
- **Web Interface**: User-friendly frontend for symptom input and result visualization
- **REST API**: FastAPI backend for integration with other systems

## Architecture

