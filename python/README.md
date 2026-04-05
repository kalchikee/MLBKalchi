# MLB Oracle v4.0 — Phase 3 Python ML Pipeline

This directory contains the Python scripts for training the ML calibration layer.

## Prerequisites

```bash
pip install -r requirements.txt
```

## Step 1: Build Historical Dataset (~30 min)

Downloads 7 seasons of MLB game data (2018–2019, 2021–2025, skipping 2020 COVID season) and computes all 30 features used in live predictions.

```bash
python build_dataset.py
```

**Options:**
```bash
python build_dataset.py --resume       # Resume from checkpoint (default behavior)
python build_dataset.py --no-resume    # Start fresh, ignore checkpoint
python build_dataset.py --year 2024    # Only download a single season
python build_dataset.py --seasons 2022 2023 2024  # Specific seasons
python build_dataset.py --sleep 0.5   # Increase sleep between API calls
```

**Output:** `data/historical_features.csv`

**Note on estimated features:** Many Statcast and live features (wind, temperature, bullpen usage, DRS, catcher framing, etc.) cannot be reconstructed from the MLB Stats API schedule endpoint. These are filled with `0.0`. The model still trains well because Elo, SP quality (ERA/xFIP/SIERA/K-BB%), team offense/defense, park factors, and Pythagorean/log5 win probability are the strongest predictors.

---

## Step 2: Train Model (~2 min)

Trains a Logistic Regression with isotonic calibration using walk-forward cross-validation.

```bash
python train_model.py
```

**Options:**
```bash
python train_model.py --evaluate-only         # Just print CV metrics, don't save
python train_model.py --input path/to/data.csv  # Custom input
python train_model.py --output-dir data/model   # Custom output directory
```

**Walk-forward CV splits:**

| Split | Train | Test |
|-------|-------|------|
| 1     | 2018–2021 | 2022 |
| 2     | 2018–2022 | 2023 |
| 3     | 2018–2023 | 2024 |
| 4     | 2018–2024 | 2025 |

**Output:** Four JSON files in `data/model/`:

- `coefficients.json` — feature weights + intercept (used by TypeScript runtime)
- `scaler.json` — StandardScaler mean/std arrays
- `calibration.json` — isotonic regression thresholds (x → y mapping)
- `model_metadata.json` — training info, CV metrics, feature names

---

## Step 3: Run TypeScript Pipeline

Once the model artifacts are saved, the TypeScript pipeline auto-loads them:

```bash
npm run dev
# or
npm start
```

The pipeline log will show either:
- `Using ML model (v4.0.0)` — model loaded, calibrated probabilities used
- `Using Monte Carlo fallback` — model files not found, Phase 1 behavior

---

## Architecture

```
build_dataset.py
    → MLB Stats API (statsapi.mlb.com)
    → Computes 30 features per game (same as featureEngine.ts)
    → Elo updated sequentially, season regression applied
    → Saves data/historical_features.csv

train_model.py
    → Loads historical_features.csv
    → Walk-forward CV (no data leakage)
    → StandardScaler normalization
    → LogisticRegression(C=1.0) base model
    → CalibratedClassifierCV(method='isotonic') for Platt/isotonic calibration
    → Saves JSON artifacts to data/model/

src/models/metaModel.ts  (TypeScript)
    → Loads coefficients.json, scaler.json, calibration.json on startup
    → predict(features, mcWinProb): scales features → logit → sigmoid → isotonic interpolation
    → Falls back to MC win probability if model not loaded

src/features/marketEdge.ts  (TypeScript)
    → computeEdge(modelProb, homeML, awayML)
    → Converts American lines to vig-removed implied probabilities
    → Returns edge category: none/small/meaningful/large/extreme
```

## Expected Performance

Based on walk-forward CV on 7 seasons:
- **Accuracy:** ~55–58% (vs 50% baseline, ~53–54% for simple ML without Statcast)
- **Brier score:** ~0.235–0.245 (vs 0.250 for random)
- **High-confidence picks (>60% prob):** ~58–62% accuracy

Performance is limited by the historical data constraints (no Statcast, no bullpen usage, no weather). Live predictions benefit from all 30 features including real-time data.
