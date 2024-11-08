import streamlit as st
from streamlit_option_menu import option_menu
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

# Top navigation bar owner name
st.markdown(
    """
    <style>
    .reportview-container .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# sidebar
with st.sidebar:
    selected = option_menu(
        menu_title='HajranAI',
        options=[
            'Home',
            'PPOK Prediction',
            'Diabetes Prediction',
            'Heart Disease Prediction',
            'Recommendation',
            'About Us'
        ],
        icons=['house', 'activity', 'droplet', 'heart', 'lightbulb', 'info-circle'],
        default_index=0
    )

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

def show_recommendation():

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

# Load Model for PPOK Prediction
@st.cache_resource
def load_model():
    model_path = "../Model_Prediction/Output_Model/xgboost_model_copd.pkl"
    model = joblib.load(model_path)
    return model

def show_prediction(model):

    # Input fields in columns for better layout
    col1, col2 = st.columns(2)
    with col1:
        fev1pred = st.number_input("FEV1 PRED", min_value=0.0, max_value=150.0, step=0.1)
        age = st.number_input("Umur", min_value=0, max_value=100, step=1)
    with col2:
        mwt1best = st.number_input("MWT1 Best", min_value=120.0, max_value=800.0, step=0.1)
        sgrq = st.number_input("SGRQ", min_value=0.0, max_value=100.0, step=0.1)
        smoking = st.selectbox("Status Merokok", ("Perokok", "Bukan Perokok"))

    # Map smoking status to numerical values
    smoking_mapping = {"Perokok": 1, "Bukan Perokok": 2}
    smoking_encoded = smoking_mapping[smoking]

    if st.button("Prediksi"):
        # Prepare input data for prediction
        input_data = pd.DataFrame([[fev1pred, age, mwt1best, sgrq, smoking_encoded]], 
                                  columns=['FEV1PRED', 'AGE', 'MWT1Best', 'SGRQ', 'smoking'])
        
        # Scaling the input data
        scaler = RobustScaler()
        input_data_scaled = scaler.fit_transform(input_data)
        
        try:
            # Get prediction from the model
            predicted_label = model.predict(input_data_scaled)[0]
            hasil_prediksi = (
                "Ringan" if predicted_label == 0 else
                "Sedang" if predicted_label == 1 else
                "Berat/Sangat Berat" if predicted_label == 2 else
                "Tidak Diketahui"
            )
            # Display prediction result
            st.success(f"Hasil Prediksi: {hasil_prediksi}")
            st.session_state.prediction_data = {
                "FEV1PRED": fev1pred,
                "AGE": age,
                "MWT1Best": mwt1best,
                "SGRQ": sgrq,
                "smoking": smoking,
                "hasil_prediksi": hasil_prediksi
            }
            
            # Recommendation button
            if st.button("Dapatkan Rekomendasi"):
                st.session_state.page = 'Recommendation'
                st.experimental_rerun()
                
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

#Multipage
if selected == 'Home':
    st.markdown(
    """
    <h1 style="text-align: center; margin-bottom: 20px;">Selamat Datang di <br> Sistem Pemantauan Kesehatan</h1>
    """, unsafe_allow_html=True
    )
    st.write("Sistem ini dapat membantu Anda dalam memantau kesehatan Anda pada Penyakit Tidak Menular (PTM)")

if selected == 'PPOK Prediction':
    st.markdown(
    """
    <h1 style="text-align: center; margin-bottom: 20px;">Prediksi PPOK</h1>
    """, unsafe_allow_html=True
    )
    model = load_model()
    show_prediction(model)

if selected == 'Diabetes Prediction':
    st.markdown(
    """
    <h1 style="text-align: center; margin-bottom: 20px;">Prediksi Diabetes</h1>
    """, unsafe_allow_html=True
    )

if selected == 'Heart Disease Prediction':
    st.markdown(
    """
    <h1 style="text-align: center; margin-bottom: 20px;">Prediksi Heart Disease</h1>
    """, unsafe_allow_html=True
    )

if selected == 'Recommendation':
    st.markdown(
    """
    <h1 style="text-align: center; margin-bottom: 20px;">Rekomendasi Berbasis RAG</h1>
    """, unsafe_allow_html=True
    )
    show_recommendation()

if selected == 'About Us':
    st.markdown(
    """
    <h1 style="text-align: center; margin-bottom: 20px;">Web App Contributor</h1>
    """, unsafe_allow_html=True
    )

    # Membuat kontainer dengan 4 anggota tim
    st.markdown(
    """
    <div style="display: flex; justify-content: space-between; gap: 20px; flex-wrap: wrap;">

    <!-- Anggota Tim 1 -->
    <div style="text-align: center;">
        <img src="https://media.licdn.com/dms/image/v2/D4D03AQFJBMvHtumirA/profile-displayphoto-shrink_200_200/profile-displayphoto-shrink_200_200/0/1703086068686?e=1736380800&v=beta&t=WHc9g4rEcaP1M568_18EGA6-1XjLqDVMMVdadax93EI" 
            style="width: 100px; height: 100px; border-radius: 50%; margin-bottom: 10px;"/>
        <div style="font-weight: bold;">Biliarto Sastro C.</div>
        <div>PM (Project Manager)</div>
    </div>

    <!-- Anggota Tim 2 -->
    <div style="text-align: center;">
        <img src="https://media.licdn.com/dms/image/v2/D5603AQEMO89szB8zUg/profile-displayphoto-shrink_200_200/profile-displayphoto-shrink_200_200/0/1703159430457?e=1736380800&v=beta&t=0xUwT4ewneeQi8p01dU7NpKoHK8oOH4jjgnykY-tFfY" 
            style="width: 100px; height: 100px; border-radius: 50%; margin-bottom: 10px;"/>
        <div style="font-weight: bold;">Hajran Azbytama W.</div>
        <div>DA (Data Analyst)</div>
    </div>

    <!-- Anggota Tim 3 -->
    <div style="text-align: center;">
        <img src="https://media.licdn.com/dms/image/v2/D5603AQFWHvsYJ9voEQ/profile-displayphoto-shrink_200_200/profile-displayphoto-shrink_200_200/0/1672065689708?e=1736380800&v=beta&t=ur9zQ-LEIAQpGS7nP41KzFURjKmexG3uHFfITbzgIr8" 
            style="width: 100px; height: 100px; border-radius: 50%; margin-bottom: 10px;"/>
        <div style="font-weight: bold;">Muhammad Goldy W. H.</div>
        <div>DS (Data Scientist)</div>
    </div>

    <!-- Anggota Tim 4 -->
    <div style="text-align: center;">
        <img src="https://media.licdn.com/dms/image/v2/D5603AQFWHvsYJ9voEQ/profile-displayphoto-shrink_200_200/profile-displayphoto-shrink_200_200/0/1672065689708?e=1736380800&v=beta&t=ur9zQ-LEIAQpGS7nP41KzFURjKmexG3uHFfITbzgIr8" 
            style="width: 100px; height: 100px; border-radius: 50%; margin-bottom: 10px;"/>
        <div style="font-weight: bold;">Rizky Anugrah</div>
        <div>SD (Software Developer)</div>
    </div>

    </div>
    """, unsafe_allow_html=True
    )