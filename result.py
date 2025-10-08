import gradio as gr
import pandas as pd
import numpy as np
import re
import joblib
from sentence_transformers import SentenceTransformer, util

# --------------------------------------------------
# Load saved models
# --------------------------------------------------
tfidf = joblib.load("tfidf_vectorizer.joblib")
model = joblib.load("tfidf_model.joblib")
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9+\-*/%()., ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def is_math_question(q):
    return bool(re.search(r"\d+|percent|ratio|speed|distance|time|sum|average|mean|profit|loss|cost|simple|interest", q))

def symbolic_solver(question):
    try:
        expr = re.findall(r"[0-9+\-*/().% ]+", question)
        if expr:
            return eval(expr[0])
    except:
        return None

def semantic_reasoning(question, options):
    q_emb = encoder.encode(question, convert_to_tensor=True)
    opts_emb = encoder.encode(options, convert_to_tensor=True)
    sims = util.cos_sim(q_emb, opts_emb).cpu().numpy()[0]
    return np.argmax(sims) + 1  # 1-based index

# --------------------------------------------------
# Main prediction function
# --------------------------------------------------
def predict_answer(problem, option1, option2, option3, option4):
    options = [option1, option2, option3, option4]
    text = clean_text(problem + " " + " ".join(options))
    
    # --- ML prediction ---
    X = tfidf.transform([text])
    ml_pred = int(model.predict(X)[0])
    
    # --- Semantic prediction ---
    sem_pred = semantic_reasoning(problem, options)
    
    # --- Symbolic prediction ---
    sym_pred = sem_pred
    if is_math_question(problem.lower()):
        val = symbolic_solver(problem)
        if val is not None:
            nums = []
            for o in options:
                m = re.sub("[^0-9.\-+]", "", str(o))
                nums.append(float(m) if m else 0)
            idx = int(np.argmin([abs(v - val) for v in nums])) + 1
            sym_pred = idx
    
    # --- Ensemble voting ---
    votes = [ml_pred, sem_pred, sym_pred]
    final = max(set(votes), key=votes.count)
    
    explanation = f"""
    🧠 **Reasoning Breakdown:**
    - TF-IDF + Logistic Regression → Option {ml_pred}
    - SentenceTransformer Semantic Similarity → Option {sem_pred}
    - Symbolic Math Layer → Option {sym_pred}
    
    ✅ **Final Voted Answer: Option {final}**
    """
    return explanation

# --------------------------------------------------
# Gradio UI
# --------------------------------------------------
iface = gr.Interface(
    fn=predict_answer,
    inputs=[
        gr.Textbox(label="Problem Statement", lines=3, placeholder="Enter your question..."),
        gr.Textbox(label="Option 1"),
        gr.Textbox(label="Option 2"),
        gr.Textbox(label="Option 3"),
        gr.Textbox(label="Option 4")
    ],
    outputs=gr.Markdown(label="Result"),
    title="🧩 Agentic Reasoning System",
    description="Hybrid Symbolic + ML Pipeline combining TF-IDF, Semantic Similarity, and Symbolic Math Reasoning.",
    theme="soft",
    examples=[
        ["What is 5 + 3?", "6", "7", "8", "9"],
        ["The average of 10 and 20 is?", "10", "15", "20", "25"],
        ["The synonym of 'happy' is?", "sad", "joyful", "angry", "bored"]
    ]
)

if __name__ == "__main__":
    iface.launch(share=True)
