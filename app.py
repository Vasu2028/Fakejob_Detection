import streamlit as st
import pickle

# Load trained model and vectorizer
with open("fake_job_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

# App title
st.title("🚨 Fake Job Detection App")
st.write("Paste a job description below and check if it's **Fake** or **Real**.")

# User input
job_text = st.text_area("Enter Job Posting Text:")

if st.button("Predict"):
    if job_text.strip() == "":
        st.warning("⚠️ Please enter some job description text!")
    else:
        # Transform input using saved TF-IDF vectorizer
        job_features = tfidf.transform([job_text])

        # Predict with model
        pred = model.predict(job_features)[0]
        proba = model.predict_proba(job_features)[0]

        # Display result
        if pred == 1:
            st.error(f"⚠️ Fake Job Posting Detected! (Confidence: {proba[1]*100:.2f}%)")
        else:
            st.success(f"✅ Real Job Posting Detected! (Confidence: {proba[0]*100:.2f}%)")
