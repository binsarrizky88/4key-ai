
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Konfigurasi koneksi ke Google Sheets
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
client = gspread.authorize(credentials)

# URL Google Sheet kamu
sheet_url = "https://docs.google.com/spreadsheets/d/1xV6db4tNF6RxAn8KaO9qqFtcd0IigBfqOeA9JJjSdhA/edit#gid=0"
sheet = client.open_by_url(sheet_url).sheet1
data = sheet.get_all_records()
df = pd.DataFrame(data)

# Load model AI
model = SentenceTransformer('all-MiniLM-L6-v2')

@st.cache(allow_output_mutation=True)
def get_embeddings():
    return model.encode(df["Trigger (Pertanyaan Member)"].tolist())

embeddings = get_embeddings()

# UI
st.title("🤖 4key-ai - Auto CS Assistant")
st.write("Tempel pertanyaan dari member, sistem akan cari balasan paling cocok.")

user_input = st.text_area("Pertanyaan Member:")

if user_input:
    input_embedding = model.encode([user_input])
    similarity = cosine_similarity(input_embedding, embeddings)[0]
    best_idx = similarity.argmax()
    st.markdown("### ✉️ Jawaban:")
    st.write(df.iloc[best_idx]["Jawaban CS"])
    st.caption(f"Skor kemiripan: {similarity[best_idx]:.2f}")
