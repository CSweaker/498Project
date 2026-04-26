"""Streamlit demo: clean -> adversarial attack -> diffusion purification -> prediction."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

from adversarial_attacks import fgsm_attack, pgd_attack
from baseline_model import load_classifier
from diffusion_purification import load_score_model, purify

st.set_page_config(page_title="Traffic Purification Demo", layout="wide")
st.title("🔐 Robust Encrypted Traffic Analysis")
st.markdown("**Generative Purification Framework Using Diffusion Models**")
st.divider()

DATA_DIR = Path("data")
ARTIFACT_DIR = Path("artifacts")
MODEL_DIR = Path("models")


@st.cache_resource
def load_artifacts():
    device = torch.device("cpu")
    X_test = np.load(DATA_DIR / "X_test.npy").astype(np.float32)
    y_test = np.load(DATA_DIR / "y_test.npy").astype(np.int64)
    preprocess = joblib.load(ARTIFACT_DIR / "preprocess.joblib")
    classifier = load_classifier(str(MODEL_DIR / "tabular_baseline.pth"), device=device)
    score_model, sigmas, _ = load_score_model(str(MODEL_DIR / "score_net.pth"), device=device)
    return classifier, score_model, sigmas, X_test, y_test, preprocess


def predict(model, x_np: np.ndarray):
    with torch.no_grad():
        x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)
        logits = model(x)
        prob = torch.softmax(logits, dim=1).numpy()[0]
        pred = int(prob.argmax())
    label = "🔴 Malicious" if pred == 1 else "🟢 Benign"
    return label, prob


st.sidebar.header("⚙️ Settings")
attack_type = st.sidebar.selectbox("Attack Method", ["FGSM", "PGD"])
epsilon = st.sidebar.slider("Epsilon (ε)", 0.01, 0.50, 0.10, step=0.01)
pgd_steps = st.sidebar.slider("PGD Steps", 5, 50, 20, step=5)
sample_idx = st.sidebar.number_input("Sample Index", min_value=0, value=0, step=1)
run_button = st.sidebar.button("▶ Run Pipeline", type="primary")

if not run_button:
    st.info("Configure settings in the sidebar and click **Run Pipeline**.")
    st.markdown(
        """
        **Pipeline:**
        ```text
        Clean traffic sample
             ↓
        FGSM / PGD adversarial perturbation
             ↓
        Score-based diffusion purification
             ↓
        Tabular ResNet classifier prediction
        ```
        """
    )
    st.stop()

try:
    classifier, score_model, sigmas, X_test, y_test, preprocess = load_artifacts()
except FileNotFoundError as exc:
    st.error(
        "Missing data/model artifacts. Run preprocessing, baseline training, attack generation, "
        "and diffusion training first.\n\n" + str(exc)
    )
    st.stop()

if sample_idx >= len(X_test):
    st.error(f"Sample index is too large. Maximum index is {len(X_test) - 1}.")
    st.stop()

x_clean = X_test[int(sample_idx)]
y_true = int(y_test[int(sample_idx)])
true_label = "Malicious" if y_true == 1 else "Benign"
feature_bounds = preprocess.get("feature_bounds")

if attack_type == "FGSM":
    x_adv = fgsm_attack(
        classifier,
        x_clean[None, :],
        np.array([y_true]),
        epsilon=epsilon,
        device="cpu",
        feature_bounds=feature_bounds,
    )[0]
else:
    x_adv = pgd_attack(
        classifier,
        x_clean[None, :],
        np.array([y_true]),
        epsilon=epsilon,
        num_steps=pgd_steps,
        device="cpu",
        feature_bounds=feature_bounds,
    )[0]

x_pur = purify(
    score_model,
    x_adv[None, :],
    sigmas=sigmas,
    device="cpu",
    feature_bounds=feature_bounds,
)[0]

pred_clean, prob_clean = predict(classifier, x_clean)
pred_adv, prob_adv = predict(classifier, x_adv)
pred_pur, prob_pur = predict(classifier, x_pur)

st.subheader(f"Sample #{sample_idx} — Ground Truth: **{true_label}**")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### 1️⃣ Clean")
    st.metric("Prediction", pred_clean)
    st.metric("Malicious confidence", f"{prob_clean[1] * 100:.1f}%")
with col2:
    st.markdown(f"### 2️⃣ {attack_type} Attack")
    st.metric("Prediction", pred_adv)
    st.metric("Malicious confidence", f"{prob_adv[1] * 100:.1f}%")
with col3:
    st.markdown("### 3️⃣ Purified")
    st.metric("Prediction", pred_pur)
    st.metric("Malicious confidence", f"{prob_pur[1] * 100:.1f}%")

st.divider()
st.subheader("📊 Feature comparison")
num_features = min(40, len(x_clean))
fig = go.Figure()
fig.add_trace(go.Scatter(y=x_clean[:num_features], name="Clean"))
fig.add_trace(go.Scatter(y=x_adv[:num_features], name="Adversarial"))
fig.add_trace(go.Scatter(y=x_pur[:num_features], name="Purified"))
fig.update_layout(xaxis_title="Feature index", yaxis_title="Standardized value", height=380)
st.plotly_chart(fig, use_container_width=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Mean |attack perturbation|", f"{np.abs(x_adv - x_clean).mean():.4f}")
with c2:
    st.metric("Mean |purified - clean|", f"{np.abs(x_pur - x_clean).mean():.4f}")
with c3:
    st.metric("Mean |purified - adversarial|", f"{np.abs(x_pur - x_adv).mean():.4f}")
