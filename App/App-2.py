import streamlit as st
import os
import glob
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema import HumanMessage
from dotenv import load_dotenv

# Initialize session state for navigation
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# Sidebar navigation
st.sidebar.title('HajranAI')

# Navigation buttons
if st.sidebar.button('🏠 Home'):
    st.session_state.page = 'Home'
if st.sidebar.button('🔍 Prediction'):
    st.session_state.page = 'Prediction'
if st.sidebar.button('💡 Recommendation'):
    st.session_state.page = 'Recommendation'
if st.sidebar.button('📖 About Us'):
    st.session_state.page = 'About'

# Recommendation System Functions
@st.cache_resource
def init_recommendation():
    # Load all PDFs from the specified folder
    pdf_folder_path = "../Data/"
    all_pdf_paths = glob.glob(os.path.join(pdf_folder_path, "*.pdf"))
    
    documents = []
    for pdf_path in all_pdf_paths:
        loader = PyPDFLoader(pdf_path)
        pdf_docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        documents.extend(text_splitter.split_documents(pdf_docs))
    
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)
    
    vector_db = FAISS.from_documents(documents, embeddings)
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    return llm, retriever

# Page Functions
def show_home():
    # Menggunakan HTML untuk mengatur perataan
    st.markdown("<h1 style='text-align: center;'>Selamat Datang di Sistem Pemantauan Kesehatan</h1>", unsafe_allow_html=True)
    st.markdown("""
    <p style='text-align: center;'>
    Penyakit Paru Obstruktif Kronik (PPOK) adalah penyakit paru-paru yang serius 
    yang menghalangi aliran udara dan membuatnya sulit untuk bernapas. 
    PPOK biasanya disebabkan oleh paparan jangka panjang terhadap iritasi paru 
    (seperti asap rokok) dan dapat menyebabkan gejala seperti batuk, 
    sesak napas, dan kelelahan.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center;'>Fakta Menarik tentang PPOK:</h2>", unsafe_allow_html=True)
    st.write("""
    - Prevalensi Tinggi: Sekitar 250 juta orang di seluruh dunia menderita PPOK.
    - Penyebab Utama: Merokok adalah penyebab utama PPOK, tetapi paparan polusi 
      udara, debu, dan bahan kimia juga berkontribusi.
    - Deteksi Dini: Tes fungsi paru-paru dapat membantu mendeteksi PPOK 
      lebih awal, bahkan sebelum gejala muncul.
    - Pengelolaan: Dengan perawatan yang tepat, orang dengan PPOK dapat 
      menjalani hidup yang aktif dan sehat.
    """)

    st.markdown("<h2 style='text-align: center;'>Data PPOK</h2>", unsafe_allow_html=True)
    st.write("""
    Menurut data dari WHO, PPOK adalah penyebab kematian ketiga terbesar 
    di dunia setelah penyakit jantung dan stroke. Pencegahan melalui 
    penghindaran rokok dan pengelolaan lingkungan dapat mengurangi 
    risiko pengembangan PPOK.
    """)

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

def show_prediction():
    st.title("Prediksi Risiko PPOK")
    st.write("Prediction in process")

def show_recommendation():
    st.title("Sistem Rekomendasi Kesehatan")
    
    llm, retriever = init_recommendation()
    
    profil_pasien = st.text_input("Masukkan Profil Pasien (umur, jenis kelamin, dll):")
    riwayat_pasien = st.text_area("Masukkan Riwayat Pasien:")
    pola_hidup = st.text_area("Masukkan Pola Hidup Pasien:")
    hasil_pred = st.selectbox("Masukkan Hasil Prediksi", ("PPOK", "Tidak PPOK"))
    
    recommendation_type = st.radio("Pilih Jenis Rekomendasi:",
                                 ("Rekomendasi Pengobatan",
                                  "Rekomendasi Pola Hidup",
                                  "Rekomendasi Penanganan Lanjutan"))
    
    if st.button("Dapatkan Rekomendasi"):
        query = f"Profil pasien: {profil_pasien}. Riwayat: {riwayat_pasien}. Pola Hidup: {pola_hidup}. Hasil Prediksi: {hasil_pred}."
        context = "\n".join([result.page_content for result in retriever.get_relevant_documents(query)])
        
        if recommendation_type == "Rekomendasi Pengobatan":
            prompt = generate_treatment_prompt(query=query, context=context)
        elif recommendation_type == "Rekomendasi Pola Hidup":
            prompt = generate_lifestyle_prompt(query=query, context=context)
        else:
            prompt = generate_followup_prompt(query=query, context=context)
        
        messages = [HumanMessage(content=prompt)]
        answer = llm(messages=messages)
        st.markdown(answer.content)

def show_about():
    st.title("Tentang Kami")
    st.write("""
    Sistem Pemantauan Kesehatan adalah platform yang menggabungkan kecerdasan buatan 
    dan pengetahuan medis untuk membantu dalam pemantauan dan pengelolaan kesehatan.
    
    Tim kami terdiri dari:
    - Pengembang Aplikasi
    - Ahli Kesehatan
    - Spesialis Data
    
    Hubungi kami di: health@example.com
    """)

# Main App Logic
def main():
    if st.session_state.page == 'Home':
        show_home()
    elif st.session_state.page == 'Prediction':
        show_prediction()
    elif st.session_state.page == 'Recommendation':
        show_recommendation()
    elif st.session_state.page == 'About':
        show_about()

if __name__ == "__main__":
    main()