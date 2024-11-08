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
from home import home_section
from about_us import about_us_section

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title='HealthAI',
        options=[
            'Home',
            'HT Prediction',
            'DM Prediction',
            'Stroke Prediction',
            'Recommendation',
            'About Us'
        ],
        icons=['house', 'activity', 'droplet', 'heart', 'lightbulb', 'info-circle'],
        default_index=0
    )

# Helper function to load model
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

# Load models
model_ht = load_model("../Model_Prediction/Output_Model/xgboost_ht.pkl")
model_dm = load_model("../Model_Prediction/Output_Model/xgboost_dm.pkl")
model_stroke = load_model("../Model_Prediction/Output_Model/xgboost_st.pkl")

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

# Fungsi untuk menampilkan input dalam tiga kolom samping
def triple_column_input(inputs):
    col1, col2, col3 = st.columns(3)
    for i, (key, value) in enumerate(inputs.items()):
        col = [col1, col2, col3][i % 3]  # Rotates among the three columns
        with col:
            if value['type'] == 'slider':
                st.slider(label=value['label'], min_value=value['min_value'], max_value=value['max_value'], step=value['step'], key=key)
            elif value['type'] == 'selectbox':
                st.selectbox(label=value['label'], options=value['options'], format_func=value['format_func'], key=key)
            elif value['type'] == 'number_input':
                st.number_input(label=value['label'], min_value=value['min_value'], max_value=value['max_value'], step=value['step'], key=key)

# Fungsi prediksi HT
def predict_ht():
    st.title("HT Prediction")
    inputs = {
        "cp": {"label": "Tipe Sakit Data", "options": [0, 1, 2, 3], "type": "selectbox", 
               "format_func": lambda x: {0: "Asymptomatic", 1: "Typical Angina", 2: "Atypical Angina", 3: "Non-Anginal"}[x]},
        "trestbps": {"label": "Trestbps", "min_value": 50.0, "max_value": 200.0, "step": 0.1, "type": "number_input"},
        "chol": {"label": "Serum cholestoral dalam mg/dl", "min_value": 100.0, "max_value": 600.0, "step": 0.1, "type": "number_input"},
        "restecg": {"label": "Hasil Resting ECG", "options": [0, 1], "type": "selectbox", 
                    "format_func": lambda x: "Normal" if x == 0 else "Abnormal"},
        "thalach": {"label": "Thalach", "min_value": 50.0, "max_value": 250.0, "step": 0.1, "type": "number_input"},
        "exang": {"label": "Latihan Selama angina", "options": [0, 1], "type": "selectbox", 
                  "format_func": lambda x: "Tidak" if x == 0 else "Ya"},
        "oldpeak": {"label": "ST Depression", "min_value": 0.0, "max_value": 10.0, "step": 0.1, "type": "number_input"},
        "slope": {"label": "Kondisi Kesehatan Umum", "options": [0, 1, 2], "type": "selectbox", 
                  "format_func": lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x]},
        "ca": {"label": "Jumlah Vessels Utama", "options": [0, 1, 2, 3, 4], "type": "selectbox", 
               "format_func": lambda x: str(x)},
        "thal": {"label": "Thal", "options": [0, 1, 2], "type": "selectbox", 
                 "format_func": lambda x: {0: "Normal", 1: "Fixed Defect", 2: "Reversable Defect"}[x]}
    }
    triple_column_input(inputs)

    if st.button("Prediksi"):
        # Menangkap input pengguna
        cp = st.session_state.cp
        trestbps = st.session_state.trestbps
        chol = st.session_state.chol
        restecg = st.session_state.restecg
        thalach = st.session_state.thalach
        exang = st.session_state.exang
        oldpeak = st.session_state.oldpeak
        slope = st.session_state.slope
        ca = st.session_state.ca
        thal = st.session_state.thal
        
        # Membuat DataFrame untuk input
        input_data = pd.DataFrame([[cp, trestbps, chol, restecg, thalach, exang, oldpeak, slope, ca, thal]], 
                                  columns=['cp', 'trestbps', 'chol', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
        
        # Lakukan scaling pada data
        scaler = RobustScaler()
        input_data_scaled = scaler.fit_transform(input_data)
        
        # Prediksi menggunakan model (model harus sudah terdefinisi)
        predicted_label = model_ht.predict(input_data_scaled)[0]
        hasil_prediksi = "Negatif" if predicted_label == 0 else "Positive"
        st.write(f"### Hasil Prediksi: {hasil_prediksi}")
        
        # Menyimpan hasil prediksi di session_state untuk halaman rekomendasi
        st.session_state['ht_prediction'] = {
            "input_data": input_data,
            "hasil_prediksi": hasil_prediksi
        }
        
        # Arahkan pengguna ke halaman rekomendasi
        if st.button("Dapatkan Rekomendasi"):
            st.session_state.page = 'Recommendation'
            st.rerun()

# Fungsi prediksi DM
def predict_dm():
    st.title("DM Prediction")
    inputs = {
        "Age": {"label": "Umur", "min_value": 0, "max_value": 100, "step": 1, "type": "number_input"},
        "HighCol": {"label": "Tinggi Kolestrol", "options": [0, 1], "type": "selectbox", 
                    "format_func": lambda x: "Tidak" if x == 0 else "Ya"},
        "BMI": {"label": "BMI", "min_value": 10.0, "max_value": 50.0, "step": 0.1, "type": "number_input"},
        "GenHlh": {"label": "Kondisi Kesehatan Umum", "options": [1, 2, 3, 4, 5], "type": "selectbox", 
                   "format_func": lambda x: {1: "Sangat Buruk", 2: "Buruk", 3: "Sedang", 4: "Baik", 5: "Sangat Baik"}[x]},
        "HighBP": {"label": "Tekanan Darah Tinggi", "options": [0, 1], "type": "selectbox", 
                   "format_func": lambda x: "Tidak" if x == 0 else "Ya"}
    }
    triple_column_input(inputs)

    if st.button("Prediksi"):
        # Menangkap input pengguna
        Age = st.session_state.Age
        HighCol = st.session_state.HighCol
        BMI = st.session_state.BMI
        GenHlh = st.session_state.GenHlh
        HighBP = st.session_state.HighBP
        
        # Membuat DataFrame untuk input
        input_data = pd.DataFrame([[Age, HighCol, BMI, GenHlh, HighBP]], 
                                  columns=['Age', 'HighCol', 'BMI', 'GenHlh', 'HighBP'])
        
        # Lakukan scaling pada data
        scaler = RobustScaler()
        input_data_scaled = scaler.fit_transform(input_data)
        
        # Prediksi menggunakan model (model harus sudah terdefinisi)
        predicted_label = model_dm.predict(input_data_scaled)[0]
        hasil_prediksi = "Negatif" if predicted_label == 0 else "Positive"
        st.write(f"### Hasil Prediksi: {hasil_prediksi}")
        
        # Menyimpan hasil prediksi di session_state untuk halaman rekomendasi
        st.session_state['dm_prediction'] = {
            "input_data": input_data,
            "hasil_prediksi": hasil_prediksi
        }
        
        # Arahkan pengguna ke halaman rekomendasi
        if st.button("Dapatkan Rekomendasi"):
            st.session_state.page = 'Recommendation'
            st.rerun()

# Fungsi prediksi Stroke
def predict_stroke():
    st.title("Stroke Prediction")
    inputs = {
        "hypertension": {"label": "Hipertensi", "options": [0, 1], "type": "selectbox", 
                         "format_func": lambda x: "Tidak" if x == 0 else "Ya"},
        "heart_disease": {"label": "Penyakit Jantung", "options": [0, 1], "type": "selectbox", 
                          "format_func": lambda x: "Tidak" if x == 0 else "Ya"},
        "ever_married": {"label": "Pernikahan", "options": [0, 1], "type": "selectbox", 
                         "format_func": lambda x: "Tidak" if x == 0 else "Ya"},
        "work_type": {"label": "Tipe Pekerjaan", "options": [0, 1, 2, 3, 4], "type": "selectbox", 
                      "format_func": lambda x: {0: "Tidak Bekerja", 1: "Anak-Anak", 2: "Govt Job", 3: "Self Employed", 4: "Private"}[x]},
        "avg_glucose_level": {"label": "Rata-Rata Glukosa", "min_value": 50.0, "max_value": 300.0, "step": 0.1, "type": "number_input"},
        "bmi": {"label": "BMI", "min_value": 10.0, "max_value": 50.0, "step": 0.1, "type": "number_input"}
    }
    triple_column_input(inputs)
    
    if st.button("Prediksi"):
        # Menangkap input pengguna
        hypertension = st.session_state.hypertension
        heart_disease = st.session_state.heart_disease
        ever_married = st.session_state.ever_married
        work_type = st.session_state.work_type
        avg_glucose_level = st.session_state.avg_glucose_level
        bmi = st.session_state.bmi
        
        # Membuat DataFrame untuk input
        input_data = pd.DataFrame([[hypertension, heart_disease, ever_married, work_type, avg_glucose_level, bmi]], 
                                  columns=['hypertension', 'heart_disease', 'ever_married', 'work_type', 'avg_glucose_level', 'bmi'])
        
        # Lakukan scaling pada data
        scaler = RobustScaler()
        input_data_scaled = scaler.fit_transform(input_data)
        
        # Prediksi menggunakan model (model harus sudah terdefinisi)
        predicted_label = model_dm.predict(input_data_scaled)[0]
        hasil_prediksi = "Negatif" if predicted_label == 0 else "Positive"
        st.write(f"### Hasil Prediksi: {hasil_prediksi}")
        
        # Menyimpan hasil prediksi di session_state untuk halaman rekomendasi
        st.session_state['st_prediction'] = {
            "input_data": input_data,
            "hasil_prediksi": hasil_prediksi
        }
        
        # Arahkan pengguna ke halaman rekomendasi
        if st.button("Dapatkan Rekomendasi"):
            st.session_state.page = 'Recommendation'
            st.rerun()

# Fungsi untuk menampilkan rekomendasi
def show_recommendation():
    # Inisialisasi LLM dan retriever
    llm, retriever = init_recommendation()

    st.title("Rekomendasi Berdasarkan Prediksi")
    predictions = {
        'ht_prediction': 'Hipertensi',
        'dm_prediction': 'Diabetes',
        'stroke_prediction': 'Stroke'
    }
    available_predictions = {key: val for key, val in predictions.items() if key in st.session_state}

    if available_predictions:
        selected_disease = st.selectbox("Pilih Penyakit untuk Rekomendasi:", list(available_predictions.values()))
        prediction_key = list(available_predictions.keys())[list(available_predictions.values()).index(selected_disease)]
        pred_data = st.session_state[prediction_key]
        
        st.write(f"### Rekomendasi untuk {selected_disease}")
        st.write("Data Input:", pred_data['input_data'])
        st.write("Hasil Prediksi:", pred_data['hasil_prediksi'])
        additional_info = st.text_area("Informasi Tambahan")
        query = f"{pred_data} {additional_info}"
        
        # Mengambil dokumen yang relevan dengan retriever
        relevant_documents = retriever.get_relevant_documents(query)
        # Menyusun konteks dari dokumen relevan yang didapatkan
        context = "\n".join([result.page_content for result in relevant_documents])
        recommendation_type = st.radio("Pilih Jenis Rekomendasi:", 
                                       ("Rekomendasi Pengobatan", "Rekomendasi Pola Hidup Sehat", "Rekomendasi Tindak Lanjut"))
        
        if st.button("Dapatkan Rekomendasi"):

            # Mengambil dokumen yang relevan dengan retriever
            relevant_documents = retriever.get_relevant_documents(query)
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
    else:
        st.write("Silakan lakukan prediksi terlebih dahulu untuk mendapatkan rekomendasi.")

# Multipage logic
def main():
    if selected == 'Home':
        home_section()
    elif selected == 'HT Prediction':
        predict_ht()
    elif selected == 'DM Prediction':
        predict_dm()
    elif selected == 'Stroke Prediction':
        predict_stroke()
    elif selected == 'Recommendation':
        show_recommendation()
    elif selected == 'About Us':
        about_us_section()

if __name__ == "__main__":
    main()