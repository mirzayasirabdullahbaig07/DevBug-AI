# app.py
import streamlit as st
import numpy as np
from joblib import load

# ----------------------------
# Load models & encoders
# ----------------------------
bug_model = load("bug_classifier.pkl")
tfidf = load("tfidf_vectorizer.pkl")
bug_encoder = load("bug_label_encoder.pkl")

dev_model = load("dev_model.pkl")
dev_encoder = load("dev_encoder.pkl")
tech_encoder = load("tech_encoder.pkl")
domain_encoder = load("domain_encoder.pkl")

from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# Unified prediction function
# ----------------------------
def predict_bug_and_developer(
    bug_text: str,
    severity: str,
    environment: str,
    error_code: int,
    tech_stack: str,
    bug_domain: str
):
    # ---------- Encode metadata ----------
    severity_map = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    env_map = {"development": 0, "staging": 1, "production": 2}

    severity_code = severity_map.get(severity.lower(), 1)  # default Medium
    env_code = env_map.get(environment.lower(), 0)         # default Development

    # Handle unseen tech/domain
    if tech_stack in tech_encoder.classes_:
        tech_code = tech_encoder.transform([tech_stack])[0]
    else:
        tech_code = tech_encoder.transform(["Other"])[0]

    if bug_domain in domain_encoder.classes_:
        domain_code = domain_encoder.transform([bug_domain])[0]
    else:
        domain_code = domain_encoder.transform(["Other"])[0]

    # ---------- Bug category prediction ----------
    bug_vec = tfidf.transform([bug_text.lower()])
    bug_label = bug_model.predict(bug_vec)[0]
    bug_category = bug_encoder.inverse_transform([bug_label])[0]

    # ---------- Developer prediction ----------
    embedding = embed_model.encode([bug_text])
    X_dev_input = np.hstack([
        embedding,
        [[severity_code, env_code, error_code, tech_code, domain_code, bug_label]]
    ])

    dev_label = dev_model.predict(X_dev_input)[0]
    developer = dev_encoder.inverse_transform([dev_label])[0]

    # Optionally: top-3 developers by probability
    probs = dev_model.predict_proba(X_dev_input)[0]
    top3_idx = np.argsort(probs)[::-1][:3]
    top3_devs = [(dev_encoder.inverse_transform([i])[0], probs[i]) for i in top3_idx]

    return bug_category, developer, top3_devs

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🪲 Bug Classifier & Developer Recommender")

st.header("Enter Bug Details")

bug_text = st.text_area("Bug Description / Title", height=120)
severity = st.selectbox("Severity", ["Low", "Medium", "High", "Critical"])
environment = st.selectbox("Environment", ["Development", "Staging", "Production"])
error_code = st.number_input("Error Code", min_value=0, step=1, value=0)
tech_stack = st.selectbox(
    "Tech Stack",
    list(tech_encoder.classes_) + ["Other"]
)
bug_domain = st.selectbox(
    "Bug Domain",
    list(domain_encoder.classes_) + ["Other"]
)

if st.button("Predict"):
    if not bug_text.strip():
        st.warning("Please enter a bug description or title.")
    else:
        bug_cat, dev, top3 = predict_bug_and_developer(
            bug_text, severity, environment, error_code, tech_stack, bug_domain
        )

        st.subheader("Predictions")
        st.markdown(f"**Bug Category:** {bug_cat}")
        st.markdown(f"**Recommended Developer:** {dev}")

        st.markdown("**Top 3 Developers (with probability):**")
        for d, p in top3:
            st.markdown(f"- {d}: {p:.2f}")

