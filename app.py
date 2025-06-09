
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Ganti link CSV di bawah dengan link sheet kamu dalam format CSV
csv_url = "https://docs.google.com/spreadsheets/d/1xV6db4tNF6RxAn8KaO9qqFtcd0IigBfqOeA9JJjSdhA/export?format=csv"

# Load data dari Google Sheets (public CSV)
df = pd.read_csv(csv_url)

# Load model AI
model = SentenceTransformer('all-MiniLM-L6-v2')

@st.cache(allow_output_mutation=True)
def get_embeddings():
    return model.encode(df["Trigger (Pertanyaan Member)"].astype(str).tolist())

embeddings = get_embeddings()

# UI
st.title("🤖 4key-ai - Auto CS Assistant (Lite Version)")
st.write("Tempel pertanyaan dari member, sistem akan cari balasan paling cocok dari Google Sheet kamu.")

user_input = st.text_area("Pertanyaan Member:")

if user_input:
    input_embedding = model.encode([user_input])
    similarity = cosine_similarity(input_embedding, embeddings)[0]
    best_idx = similarity.argmax()
    st.markdown("### ✉️ Jawaban:")
    st.write(df.iloc[best_idx]["Jawaban CS"])
    st.caption(f"Skor kemiripan: {similarity[best_idx]:.2f}")
