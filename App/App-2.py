import streamlit as st
import os
import glob
import numpy as np
import pandas as pd
import joblib
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema import HumanMessage
from dotenv import load_dotenv
from sklearn.preprocessing import RobustScaler

# Initialize session state for navigation and data
if 'page' not in st.session_state:
    st.session_state.page = 'Home'
if 'prediction_data' not in st.session_state:
    st.session_state.prediction_data = None

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

# Load Model for Prediction
@st.cache_resource
def load_model():
    # Load the model using joblib
    model_path = "../Model_Prediction/Output_Model/xgboost_model_copd.pkl"
    model = joblib.load(model_path)
    return model

# Define RAG prompt templates for different recommendations
def generate_treatment_prompt(query, context):
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

def generate_lifestyle_prompt(query, context):
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

def generate_followup_prompt(query, context):
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

# Page Functions
def show_home():
    st.markdown("<h1 style='text-align: center;'>Selamat Datang di Sistem Pemantauan Kesehatan</h1>", unsafe_allow_html=True)
    st.write("""
    Penyakit Paru Obstruktif Kronik (PPOK) adalah penyakit paru-paru yang serius 
    yang menghalangi aliran udara dan membuatnya sulit untuk bernapas.
    """)

def show_prediction(model):
    st.title("Prediksi Risiko PPOK")

    # Input fields for prediction
    fev1pred = st.number_input("FEV1 PRED", min_value=0.0, max_value=150.0, step=0.1)
    age = st.number_input("Umur", min_value=0, max_value=100, step=1)
    mwt1best = st.number_input("MWT1 Best", min_value=120.0, max_value=800.0, step=0.1)
    sgrq = st.number_input("SGRQ", min_value=0.0, max_value=100.0, step=0.1)
    smoking = st.selectbox("Status Merokok", ("Perokok", "Bukan Perokok"))

    # Preprocess smoking status
    smoking_mapping = {"Perokok": 1, "Bukan Perokok": 2}
    smoking_encoded = smoking_mapping[smoking]

    # model = load_model()

    if st.button("Prediksi"):
        
        # if model is not None:
        # Create input data as a DataFrame with named columns
        input_data = pd.DataFrame([[fev1pred, age, mwt1best, sgrq, smoking_encoded]], 
                                columns=['FEV1PRED', 'AGE', 'MWT1Best', 'SGRQ', 'smoking'])
        
        scaler = RobustScaler()
        input_data_scaled = scaler.fit_transform(input_data)
        input_data_reshaped = input_data_scaled.reshape(1, -1)
        
        try:
            predicted_label = model.predict(input_data_reshaped)[0]
            hasil_prediksi = (
                "Ringan" if predicted_label == 0 else
                "Sedang" if predicted_label == 1 else
                "Berat/Sangat Berat" if predicted_label == 2 else
                "Tidak Diketahui"  # Menambahkan nilai default untuk kasus lainnya
            )
            
            st.write(f"Hasil Prediksi: {hasil_prediksi}")
            
            st.session_state.prediction_data = {
                "FEV1PRED": fev1pred,
                "AGE": age,
                "MWT1Best": mwt1best,
                "SGRQ": sgrq,
                "smoking": smoking,
                "hasil_prediksi": hasil_prediksi
            }

            if st.button("Dapatkan Rekomendasi"):
                st.session_state.page = 'Recommendation'
                st.experimental_rerun()
                
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

def show_recommendation():
    st.title("Sistem Rekomendasi Kesehatan")

    if st.session_state.prediction_data:
        llm, retriever = init_recommendation()

        pred_data = st.session_state.prediction_data
        st.write(f"FEV1 PRED: {pred_data['FEV1PRED']}")
        st.write(f"Umur: {pred_data['AGE']}")
        st.write(f"MWT1 Best: {pred_data['MWT1Best']}")
        st.write(f"SGRQ: {pred_data['SGRQ']}")
        st.write(f"Status Merokok: {pred_data['smoking']}")
        st.write(f"Hasil Prediksi: {pred_data['hasil_prediksi']}")

        additional_info = st.text_area("Informasi Tambahan")

        recommendation_type = st.radio("Pilih Jenis Rekomendasi:", 
                                     ("Rekomendasi Pengobatan", 
                                      "Rekomendasi Pola Hidup Sehat", 
                                      "Rekomendasi Tindak Lanjut"))

        if st.button("Dapatkan Rekomendasi"):
            if recommendation_type == "Rekomendasi Pengobatan":
                context = additional_info
                prompt = generate_treatment_prompt(str(pred_data), context)
            elif recommendation_type == "Rekomendasi Pola Hidup Sehat":
                context = additional_info
                prompt = generate_lifestyle_prompt(str(pred_data), context)
            else:
                context = additional_info
                prompt = generate_followup_prompt(str(pred_data), context)

            # response = retriever.retriever(prompt)
            # st.write(f"Rekomendasi: {response}")
            messages = [HumanMessage(content=prompt)]
            answer = llm(messages=messages)
            st.markdown(answer.content)

def show_about():
    st.markdown("# Tentang Kami")
    st.write("""
    Sistem ini dirancang untuk membantu petugas medis memberikan rekomendasi pengobatan
    berdasarkan hasil prediksi dan data pasien, serta memberikan saran terkait pola hidup sehat.
    """)

# Display selected page
def main():
    model = load_model()
    if st.session_state.page == 'Home':
        show_home()
    elif st.session_state.page == 'Prediction':
        show_prediction(model)
    elif st.session_state.page == 'Recommendation':
        show_recommendation()
    elif st.session_state.page == 'About':
        show_about()

if __name__ == '__main__':
    main()