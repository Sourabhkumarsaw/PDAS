# AI-Assisted Personal Decision Audit System (PDAS)

PDAS is a Flask + SQLite web application for logging personal decisions and generating explainable audit reports.
It is a **decision logging and analysis system**, a **decision-support tool**, and a **risk & bias detection system**.
It is **not** a decision-maker, chatbot, prediction engine, or psychology diagnosis tool.

## Features
- User module with registration, login, decision entry, and decision history.
- Decision Analysis Engine using TF-IDF + cosine similarity, rule-based scoring, and pattern detection.
- Audit Report Generator with risk summary, detected patterns, and improvement suggestions.
- Admin module for governance-oriented rule notes and system analytics.

## Fixed risk logic
- High stress (>4) → +2 risk
- Low confidence (≤2) → +2 risk
- Repeated similar decisions → +2 risk
- Negative past outcome → +3 risk

Risk levels:
- 0–3 → Low Risk
- 4–6 → Medium Risk
- 7+ → High Risk

## Database tables
- `users`
- `decisions`
- `decision_analysis`
- `rules`
- `audit_reports`

## Tech stack
- Frontend: HTML, CSS, JavaScript
- Backend: Python (Flask)
- Database: SQLite
- AI: scikit-learn (TF-IDF + cosine similarity)

## Run locally
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000`.

### Default admin account
- Username: `admin`
- Password: `admin123`
