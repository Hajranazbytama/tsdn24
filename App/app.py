import streamlit as st
from streamlit_option_menu import option_menu
import os
import glob
import numpy as np
import pandas as pd
import joblib
#from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
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

# Fungsi untuk memuat dokumen berbasis penyakit dalam format .txt
def load_documents_by_disease(disease):
    txt_folder_path = f"../Data/{disease}"  # Folder untuk setiap penyakit
    all_txt_paths = glob.glob(os.path.join(txt_folder_path, "*.txt"))
    
    documents = []
    for txt_path in all_txt_paths:
        loader = TextLoader(txt_path)
        txt_docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        documents.extend(text_splitter.split_documents(txt_docs))
    
    return documents

 # Inisialisasi RAG berdasarkan penyakit
@st.cache_resource
def init_recommendation():
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)

    # Membuat retriever untuk setiap penyakit dengan database terpisah
    retrievers = {}
    for disease in ["HT", "DM", "Stroke"]:
        documents = load_documents_by_disease(disease)
        vector_db = FAISS.from_documents(documents, embeddings)
        retrievers[disease] = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
    return llm, retrievers   

# Prompt untuk rekomendasi pengobatan
def generate_treatment_prompt(query, context, selected_disease):
    prompt = f"""
    Anda adalah seorang ahli kesehatan yang membantu petugas kesehatan untuk memberikan rekomendasi pengobatan kepada pasien terkait penyakit {selected_disease}.

    **Profil, Riwayat Pasien, dan hasil prediksi**:
    {query}

    **Riwayat Medis dan Keterangan Medis**:
    {context}

    Berdasarkan informasi di atas, berikan rekomendasi pengobatan yang singkat namun spesifik dan jelas meliputi:
    1. Obat yang disarankan beserta dosisnya (jika memungkinkan).
    2. Metode pengobatan yang sesuai.
    3. Langkah perawatan yang harus dilakukan oleh petugas medis terhadap pasien.
    """
    return prompt

# Prompt untuk rekomendasi pola hidup
def generate_lifestyle_prompt(query, context, selected_disease):
    prompt = f"""
    Anda adalah seorang ahli kesehatan yang membantu petugas kesehatan untuk memberikan rekomendasi pola hidup kepada pasien terkait penyakit {selected_disease}.

    **Profil, Riwayat Pasien, dan hasil prediksi**:
    {query}

    **Riwayat Medis dan Keterangan Medis**:
    {context}

    Berdasarkan informasi di atas, berikan rekomendasi pola hidup yang singkat namun spesifik dan jelas meliputi:
    1. Pola makan yang disarankan.
    2. Aktivitas fisik atau latihan yang direkomendasikan.
    3. Kebiasaan atau gaya hidup yang perlu dihindari.
    """
    return prompt

# Prompt untuk rekomendasi penanganan lanjutan
def generate_followup_prompt(query, context, selected_disease):
    prompt = f"""
    Anda adalah seorang ahli kesehatan yang membantu petugas kesehatan untuk memberikan rekomendasi penanganan lanjutan kepada pasien terkait penyakit {selected_disease}.

    **Profil, Riwayat Pasien, dan hasil prediksi**:
    {query}

    **Riwayat Medis dan Keterangan Medis**:
    {context}

    Berdasarkan informasi di atas, berikan rekomendasi penanganan lanjutan yang singkat namun spesifik dan jelas meliputi:
    1. Tindak lanjut medis yang perlu dilakukan oleh pasien atau petugas kesehatan.
    2. Jadwal kunjungan atau pemeriksaan ulang yang dianjurkan.
    3. Tes atau pemeriksaan tambahan yang disarankan (jika ada).
    """
    return prompt

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
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
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
        "GenHlth": {"label": "Kondisi Kesehatan Umum", "options": [1, 2, 3, 4, 5], "type": "selectbox", 
                   "format_func": lambda x: {1: "Sangat Buruk", 2: "Buruk", 3: "Sedang", 4: "Baik", 5: "Sangat Baik"}[x]},
        "HighBP": {"label": "Tekanan Darah Tinggi", "options": [0, 1], "type": "selectbox", 
                   "format_func": lambda x: "Tidak" if x == 0 else "Ya"}
    }
    triple_column_input(inputs)

    if st.button("Prediksi"):
        # Menangkap input pengguna
        Age = st.session_state.Age
        HighCol = st.session_state.HighCol
        Bmi = st.session_state.BMI
        GenHlth = st.session_state.GenHlth
        HighBP = st.session_state.HighBP
        
        # Membuat DataFrame untuk input
        input_data = pd.DataFrame([[Age, HighCol, Bmi, GenHlth, HighBP]], 
                                  columns=['Age', 'HighCol', 'BMI', 'GenHlth', 'HighBP'])
        
        # Lakukan scaling pada data
        scaler = RobustScaler()
        input_data_scaled = scaler.fit_transform(input_data)
        
        # Prediksi menggunakan model (model harus sudah terdefinisi)
        predicted_label = model_dm.predict(input_data_scaled)[0]
        hasil_prediksi = "Negatif" if predicted_label == 0 else "Positive"
        st.write(f"### Hasil Prediksi: {hasil_prediksi}")
        
        # Menyimpan hasil prediksi di session_state untuk halaman rekomendasi
        st.session_state['dm_prediction'] = {
            "Age": Age,
            "HighCol": HighCol,
            "BMI": Bmi,
            "GenHlth": GenHlth,
            "HighBP": HighBP,
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
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "ever_married": ever_married,
            "work_type": work_type,
            "avg_glucose_level": avg_glucose_level,
            "bmi": bmi,
            "hasil_prediksi": hasil_prediksi
        }
        
        # Arahkan pengguna ke halaman rekomendasi
        if st.button("Dapatkan Rekomendasi"):
            st.session_state.page = 'Recommendation'
            st.rerun()

# Fungsi untuk menampilkan rekomendasi dengan pemusatan menggunakan HTML
def show_recommendation():
    # Inisialisasi LLM dan retrievers
    llm, retrievers = init_recommendation()

    # Memusatkan judul
    st.markdown("<h1 style='text-align: center;'>Rekomendasi Berbasis RAG</h1>", unsafe_allow_html=True)
    
    predictions = {
        'ht_prediction': 'Hipertensi',
        'dm_prediction': 'Diabetes',
        'st_prediction': 'Stroke'
    }
    available_predictions = {key: val for key, val in predictions.items() if key in st.session_state}

    if available_predictions:
        # Memusatkan pilihan penyakit untuk rekomendasi
        selected_disease = st.selectbox("Pilih Penyakit untuk Rekomendasi:", list(available_predictions.values()), index=0, key="disease_select")
        
        # Mengambil data prediksi dari state
        prediction_key = list(available_predictions.keys())[list(available_predictions.values()).index(selected_disease)]
        pred_data = st.session_state[prediction_key]
        df = pd.DataFrame([pred_data])
        
        # Memusatkan informasi data prediksi
        st.markdown(f"<h3 style='text-align: center;'>Rekomendasi untuk {selected_disease}</h3>", unsafe_allow_html=True)
        st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
        st.dataframe(df, width=700)
        st.markdown("</div>", unsafe_allow_html=True)

        # Memusatkan area untuk input informasi tambahan
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        additional_info = st.text_area("Informasi Tambahan", height=100)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Membuat query dari data prediksi dan informasi tambahan
        query = f"{pred_data} {additional_info}"

        # Mengambil retriever sesuai penyakit yang dipilih
        disease_key = "HT" if selected_disease == "Hipertensi" else "DM" if selected_disease == "Diabetes" else "Stroke"
        retriever = retrievers[disease_key]

        # Mengambil dokumen yang relevan dengan retriever
        relevant_documents = retriever.get_relevant_documents(query)
        context = "\n".join([result.page_content for result in relevant_documents])
        
        # Memusatkan radio pilihan jenis rekomendasi
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        recommendation_type = st.radio("Pilih Jenis Rekomendasi:", 
                                       ("Rekomendasi Pengobatan", "Rekomendasi Pola Hidup Sehat", "Rekomendasi Tindak Lanjut"))
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Memusatkan tombol dan hasil rekomendasi
        if st.button("Dapatkan Rekomendasi"):
            # Menyusun prompt berdasarkan jenis rekomendasi yang dipilih
            if recommendation_type == "Rekomendasi Pengobatan":
                prompt = generate_treatment_prompt(query, context, selected_disease)
            elif recommendation_type == "Rekomendasi Pola Hidup Sehat":
                prompt = generate_lifestyle_prompt(query, context, selected_disease)
            else:
                prompt = generate_followup_prompt(query, context, selected_disease)

            # Menghasilkan jawaban menggunakan LLM dengan prompt yang telah disusun
            messages = [HumanMessage(content=prompt)]
            answer = llm(messages=messages)

            # Menampilkan jawaban rekomendasi dengan pemusatan
            st.markdown(f"<div style='text-align: center;'><strong>Rekomendasi {recommendation_type}:</strong><br>{answer.content}</div>", unsafe_allow_html=True)
    else:
        # Tampilkan pesan jika tidak ada prediksi yang tersedia
        st.markdown("<div style='text-align: center;'>Silakan lakukan prediksi terlebih dahulu untuk mendapatkan rekomendasi.</div>", unsafe_allow_html=True)

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