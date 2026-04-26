# Robust Encrypted Traffic Analysis

A reproducible course-project pipeline for adversarially robust intrusion detection on CIC-IDS2017-style flow features.

## What changed from the original version

- Fixed preprocessing leakage: the scaler is fitted only on the training split.
- Added saved preprocessing artifacts: scaler, feature list, class weights, and feature clipping bounds.
- Replaced the main baseline with a stronger ResNet-style MLP for tabular flow features while keeping a CNN class for ablation compatibility.
- Added bounded FGSM/PGD attacks for standardized tabular features.
- Added schedule-consistent score-based purification for adversarial samples.
- Updated Streamlit app imports and checkpoint loading.

## Suggested project structure

```text
498Project/
├── data/                  # put CIC-IDS2017 CSV files here; do not commit large data
├── artifacts/             # preprocessing artifact files
├── models/                # model checkpoints
├── results/               # generated metrics and adversarial arrays
├── data_preprocessing.py
├── baseline_model.py
├── adversarial_attacks.py
├── diffusion_purification.py
├── app.py
├── utils.py
├── requirements.txt
└── README.md
```

## Install

```bash
pip install -r requirements.txt
```

## Run pipeline

```bash
python data_preprocessing.py --data-dir data --out-data-dir data --out-artifact-dir artifacts
python baseline_model.py --data-dir data --artifact-dir artifacts --model-dir models --epochs 20
python adversarial_attacks.py --data-dir data --artifact-dir artifacts --model-path models/tabular_baseline.pth --out-dir results --attack both
python diffusion_purification.py --data-dir data --artifact-dir artifacts --model-dir models --baseline-path models/tabular_baseline.pth --adv-dir results --epochs 30
streamlit run app.py
```

## Notes for the report

Use four evaluation columns: clean baseline, attacked baseline, purified defense, and adaptive/defense-aware attack if time allows. For intrusion detection, report accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and attack success rate rather than accuracy alone.

## GitHub reminder

Make the repository public and add `masoud-y` as a collaborator from GitHub repository settings.
