# 🪲 DevBug-AI

**DevBug-AI** is a unified bug classification and developer recommendation system. It helps classify software bugs into categories and suggests the most suitable developers for fixing them based on historical bug data and textual descriptions.

---

## 🚀 Demo
🔗 [Live App on Streamlit](https://devbug-ai.streamlit.app/)

## 🚀 Video Demo

--- 

## 🚀 Features

- **Bug Classification:** Automatically predicts the category of a bug from its title and description.
- **Developer Recommendation:** Suggests the best developer for a bug based on metadata and text embeddings.
- **Top-3 Recommendations:** Provides top-3 developers with associated probabilities for better decision-making.
- **Streamlit Interface:** Easy-to-use web interface for entering bug details and getting predictions.
- **Handles Open Categories:** Supports unseen tech stacks or bug domains by mapping them to `"Other"`.

---

## 🧰 Tech Stack

- Python 3.9+
- [Streamlit](https://streamlit.io/) for UI
- [Scikit-learn](https://scikit-learn.org/stable/) for bug classification
- [LightGBM](https://lightgbm.readthedocs.io/en/latest/) for developer recommendation
- [Sentence Transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) for text embeddings
- [Joblib](https://joblib.readthedocs.io/) for model serialization
- Pandas & NumPy for data processing

---

## 📦 Files in the Repository

| File | Description |
|------|-------------|
| `app.py` | Streamlit web app for predictions |
| `bug.py` | Data cleaning, model training, and unified inference pipeline |
| `bug_dataset_50k.csv` | Dataset containing bug reports and developer assignments |
| `bug_inputs.xlsx` | Sample Excel file with 10 example bug reports |
| `bug_classifier.pkl` | Pre-trained bug category classifier |
| `tfidf_vectorizer.pkl` | TF-IDF vectorizer used for bug text |
| `bug_label_encoder.pkl` | Label encoder for bug categories |
| `dev_model.pkl` | Pre-trained LightGBM developer recommendation model |
| `dev_encoder.pkl` | Label encoder for developer roles |
| `tech_encoder.pkl` | Label encoder for tech stacks |
| `domain_encoder.pkl` | Label encoder for bug domains |
| `requirements.txt` | Python package dependencies |

---

## 📝 Installation

1. Clone the repository:

```bash
git clone https://github.com/mirzayasirabdullahbaig07/DevBug-AI.git
cd DevBug-AI
