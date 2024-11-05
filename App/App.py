import os
import glob
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema import HumanMessage
from dotenv import load_dotenv

# Load all PDFs from the specified folder
pdf_folder_path = "../Data/"
all_pdf_paths = glob.glob(os.path.join(pdf_folder_path, "*.pdf"))

# Load each PDF document and split text
documents = []
for pdf_path in all_pdf_paths:
    loader = PyPDFLoader(pdf_path)
    pdf_docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents.extend(text_splitter.split_documents(pdf_docs))

print(f"Total loaded document chunks: {len(documents)}")

# Set up embeddings and LLM with Google Gemini API
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)

# Create FAISS vector database from documents
vector_db = FAISS.from_documents(documents, embeddings)
retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Define RAG prompt templates for different recommendations
def generate_treatment_prompt(query, context, recommendation_type):
    prompt = f"""
    Anda adalah seorang ahli kesehatan khususnya penyakit PPOK (penyakit paru obstruktif kronik) yang membantu petugas kesehatan untuk memberikan rekomendasi pengobatan kepada pasien berdasarkan informasi yang tersedia.

    **Profil dan Riwayat Pasien**:
    {query}

    **Riwayat Medis dan Keterangan Medis**:
    {context}

    Berdasarkan informasi di atas, berikan rekomendasi pengobatan yang singkat namun spesifik dan jelas yang meliputi:
    1. Obat yang disarankan beserta dosisnya (jika mungkin).
    2. Metode pengobatan yang sesuai.
    3. Langkah perawatan yang harus dilakukan oleh petugas medis terhadap pasien.
    """
    return prompt

def generate_lifestyle_prompt(query, context, recommendation_type):
    prompt = f"""
    Anda adalah seorang ahli kesehatan khususnya penyakit PPOK (penyakit paru obstruktif kronik) yang membantu petugas kesehatan memberikan rekomendasi pola hidup sehat kepada pasien yang terindikasi TB atau tidak dengan informasi berikut.

    **Profil dan Riwayat Pasien**:
    {query}

    **Riwayat Medis dan Keterangan Medis**:
    {context}

    Berdasarkan informasi di atas, berikan rekomendasi pola hidup yang singkat namun spesifik yang mencakup:
    1. Aktivitas fisik yang aman dan direkomendasikan (misalnya, jenis olahraga dan frekuensinya).
    2. Pola makan dan jenis makanan yang sebaiknya dikonsumsi dan dihindari (contoh: makanan yang meningkatkan imunitas).
    3. Kebiasaan sehari-hari yang dapat membantu pemulihan, termasuk tips manajemen stres dan tidur.
    4. Instruksi khusus untuk menjaga kebersihan dan mencegah penularan.
    """
    return prompt

def generate_followup_prompt(query, context, recommendation_type):
    prompt = f"""
    Anda adalah seorang ahli kesehatan khususnya penyakit PPOK (penyakit paru obstruktif kronik) yang memberikan rekomendasi penanganan lanjutan bagi petugas kesehatan untuk pasien yang terindikasi TB atau tidak dengan informasi berikut.

    **Profil dan Riwayat Pasien**:
    {query}

    **Riwayat Medis dan Keterangan Medis**:
    {context}

    Berdasarkan informasi di atas, berikan rekomendasi penanganan lanjutan yang singkat namun spesifik mencakup:
    1. Jadwal kontrol kesehatan atau pemeriksaan lanjutan yang disarankan.
    2. Pengujian tambahan atau pemeriksaan yang mungkin diperlukan (contoh: X-ray atau tes laboratorium).
    3. Tanda atau gejala yang perlu diwaspadai sebagai indikasi komplikasi.
    4. Saran untuk pemulihan yang berkelanjutan, seperti adaptasi pola hidup, manajemen stres, dan dukungan sosial yang dibutuhkan.
    """
    return prompt

# Streamlit App
def main():
    st.title("Sistem Rekomendasi Kesehatan Berbasis RAG")

    # Input fields
    profil_pasien = st.text_input("Masukkan Profil Pasien (umur, jenis kelamin, dll):")
    riwayat_pasien = st.text_area("Masukkan Riwayat Pasien:")
    pola_hidup = st.text_area("Masukkan Pola Hidup Pasien:")
    hasil_ctscan = st.selectbox("Masukkan Hasil Prediksi PPOK", ("PPOK", "Tidak"))

    # Gabungkan data pasien sebagai query untuk konteks rekomendasi
    query = f"Profil pasien: {profil_pasien}." if profil_pasien else ""
    query += f" Riwayat: {riwayat_pasien}." if riwayat_pasien else ""
    query += f" Pola Hidup: {pola_hidup}." if pola_hidup else ""
    query += f" Hasil Prediksi PPOK: {hasil_ctscan}." if hasil_ctscan else ""

    # Ambil konteks dari dokumen relevan
    context_results = retriever.get_relevant_documents(query)
    
    # Ambil hanya konten dari dokumen yang relevan
    relevant_contexts = [result.page_content for result in context_results]
    context = "\n".join(relevant_contexts) if relevant_contexts else "Tidak ada konteks relevan ditemukan."

    # Buat tempat untuk menampung hasil rekomendasi
    rekomendasi_hasil = {}

    # Tombol untuk setiap jenis rekomendasi
    if st.button("Rekomendasi Pengobatan"):
        recommendation_type = "Rekomendasi Pengobatan"
        prompt = generate_treatment_prompt(query=query, context=context, recommendation_type=recommendation_type.split()[1])
        messages = [HumanMessage(content=prompt)]
        answer = llm(messages=messages)
        rekomendasi_hasil[recommendation_type] = answer

    if st.button("Rekomendasi Pola Hidup"):
        recommendation_type = "Rekomendasi Pola Hidup"
        prompt = generate_lifestyle_prompt(query=query, context=context, recommendation_type=recommendation_type.split()[1])
        messages = [HumanMessage(content=prompt)]
        answer = llm(messages=messages)
        rekomendasi_hasil[recommendation_type] = answer

    if st.button("Rekomendasi Penanganan Lanjutan"):
        recommendation_type = "Rekomendasi Penanganan Lanjutan"
        prompt = generate_followup_prompt(query=query, context=context, recommendation_type=recommendation_type.split()[1])
        messages = [HumanMessage(content=prompt)]
        answer = llm(messages=messages)
        rekomendasi_hasil[recommendation_type] = answer

    # Tampilkan hasil rekomendasi di bawah tombol
    if rekomendasi_hasil:
        st.subheader("Hasil Rekomendasi:")
        for key, value in rekomendasi_hasil.items():
            st.markdown(f"**{key}:** {value}")

    # Penanganan jika tidak ada konteks ditemukan
    if not relevant_contexts:
        st.warning("Tidak ada informasi relevan ditemukan. Silakan coba dengan input yang berbeda.")

if __name__ == "__main__":
    main()