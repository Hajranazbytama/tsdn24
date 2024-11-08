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

# Import custom pages for Home and About Us sections
from home import home_section
from about_us import about_us_section

# Sidebar menu
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

# Helper function to load PPOK prediction model
@st.cache_resource
def load_ppok_model():
    model_path = "../Model_Prediction/Output_Model/xgboost_model_copd.pkl"
    model = joblib.load(model_path)
    return model

# Initialize RAG-based recommendation system
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

# Prompt templates
def generate_treatment_prompt(query, context):
    return f"""
    Profil dan Riwayat Pasien: {query}
    Riwayat Medis dan Keterangan Medis: {context}
    Berikan rekomendasi pengobatan yang singkat namun spesifik dan jelas.
    """

def generate_lifestyle_prompt(query, context):
    return f"""
    Profil dan Riwayat Pasien: {query}
    Riwayat Medis dan Keterangan Medis: {context}
    Berikan rekomendasi pola hidup yang singkat namun spesifik.
    """

def generate_followup_prompt(query, context):
    return f"""
    Profil dan Riwayat Pasien: {query}
    Riwayat Medis dan Keterangan Medis: {context}
    Berikan rekomendasi penanganan lanjutan yang singkat namun spesifik.
    """

# Show prediction results and recommendations
def show_prediction_results(disease, prediction_data):
    st.write(f"### {disease} Prediction")
    st.write(f"FEV1 PRED: {prediction_data['FEV1PRED']}")
    st.write(f"Umur: {prediction_data['AGE']}")
    st.write(f"MWT1 Best: {prediction_data['MWT1Best']}")
    st.write(f"SGRQ: {prediction_data['SGRQ']}")
    st.write(f"Status Merokok: {prediction_data['smoking']}")
    st.write(f"Hasil Prediksi: {prediction_data['hasil_prediksi']}")

# PPOK Prediction function
def show_ppok_prediction():
    model = load_ppok_model()
    st.write("### PPOK Prediction")
    
    col1, col2 = st.columns(2)
    with col1:
        fev1pred = st.number_input("FEV1 PRED", min_value=0.0, max_value=150.0, step=0.1)
        age = st.number_input("Umur", min_value=0, max_value=100, step=1)
    with col2:
        mwt1best = st.number_input("MWT1 Best", min_value=120.0, max_value=800.0, step=0.1)
        sgrq = st.number_input("SGRQ", min_value=0.0, max_value=100.0, step=0.1)
        smoking = st.selectbox("Status Merokok", ("Perokok", "Bukan Perokok"))

    # Map smoking status to numerical values
    smoking_encoded = {"Perokok": 1, "Bukan Perokok": 2}[smoking]

    if st.button("Prediksi"):
        input_data = pd.DataFrame([[fev1pred, age, mwt1best, sgrq, smoking_encoded]], 
                                  columns=['FEV1PRED', 'AGE', 'MWT1Best', 'SGRQ', 'smoking'])
        scaler = RobustScaler()
        input_data_scaled = scaler.fit_transform(input_data)
        
        # Prediksi menggunakan model
        predicted_label = model.predict(input_data_scaled)[0]
        hasil_prediksi = (
            "Ringan" if predicted_label == 0 else
            "Sedang" if predicted_label == 1 else
            "Berat/Sangat Berat" if predicted_label == 2 else
            "Tidak Diketahui"
        )
        
        # Menyimpan hasil prediksi ke session_state
        st.session_state['ppok_prediction'] = {
            "FEV1PRED": fev1pred,
            "AGE": age,
            "MWT1Best": mwt1best,
            "SGRQ": sgrq,
            "smoking": smoking,
            "hasil_prediksi": hasil_prediksi
        }
        
        # Set session state untuk halaman rekomendasi
        st.session_state.page = 'Recommendation'
        st.rerun()

# Display recommendations based on saved prediction data
def show_recommendation():
    # Inisialisasi LLM dan retriever
    llm, retriever = init_recommendation()

    # Cek apakah ada prediksi PPOK di session_state
    if 'ppok_prediction' in st.session_state:
        st.write("### PPOK Recommendations")
        show_prediction_results("PPOK", st.session_state['ppok_prediction'])
        
        additional_info = st.text_area("Informasi Tambahan PPOK")
        recommendation_type = st.radio("Pilih Jenis Rekomendasi:", 
                                       ("Rekomendasi Pengobatan", "Rekomendasi Pola Hidup Sehat", "Rekomendasi Tindak Lanjut"))
        if st.button("Dapatkan Rekomendasi"):
            # Menyusun query untuk retriever berdasarkan prediksi PPOK
            pred_data = st.session_state['ppok_prediction']
            query = str(pred_data)

            # Mengambil dokumen yang relevan dengan retriever
            relevant_documents = retriever.get_relevant_documents(query)

            # Menyusun konteks dari dokumen relevan yang didapatkan
            context = "\n".join([result.page_content for result in relevant_documents])

            # Menyusun prompt berdasarkan jenis rekomendasi yang dipilih
            if recommendation_type == "Rekomendasi Pengobatan":
                prompt = generate_treatment_prompt(query, context)
            elif recommendation_type == "Rekomendasi Pola Hidup Sehat":
                prompt = generate_lifestyle_prompt(query, context)
            else:
                prompt = generate_followup_prompt(query, context)

            # Menghasilkan jawaban menggunakan LLM dengan prompt yang telah disusun
            messages = [HumanMessage(content=prompt)]
            answer = llm(messages=messages)
            st.markdown(f"**Rekomendasi {recommendation_type}:** {answer.content}")

    # Cek apakah ada prediksi Diabetes di session_state
    if 'diabetes_prediction' in st.session_state:
        st.write("### Diabetes Recommendations (Coming Soon)")

    # Cek apakah ada prediksi Heart Disease di session_state
    if 'heart_disease_prediction' in st.session_state:
        st.write("### Heart Disease Recommendations (Coming Soon)")

# Multipage logic
if selected == 'Home':
    home_section()
elif selected == 'PPOK Prediction':
    show_ppok_prediction()
elif selected == 'Diabetes Prediction':
    st.write("### Diabetes Prediction (Coming Soon)")
elif selected == 'Heart Disease Prediction':
    st.write("### Heart Disease Prediction (Coming Soon)")
elif selected == 'Recommendation':
    show_recommendation()
    # st.write("### Rekomendasi PTM berbasis RAG")
elif selected == 'About Us':
    about_us_section()