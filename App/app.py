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
# from sklearn.preprocessing import RobustScaler
# from sklearn.preprocessing import StandardScaler
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
            'Lung Cancer Prediction',
            'Recommendation',
            'About Us'
        ],
        icons=['house', 'activity', 'droplet', 'heart', 'lightbulb', 'info-circle'],
        default_index=0
    )

@st.cache_resource
def load_model(model_path):
# Helper function to load model
    return joblib.load(model_path)

# Load models
model_ht = load_model("../Output Model/model_ht.pkl")
model_dm = load_model("./Output Model/model_dm.pkl")
model_lc= load_model("./Output Model/model_lc.pkl")

@st.cache_resource
def load_scaler(scaler_path):
# Helper function to load scaler
    return joblib.load(scaler_path)

# Load scaler
scaler_ht = load_scaler("./Output_Model/scaler_ht.pkl")
scaler_dm = load_scaler("./Output_Model/scaler_dm.pkl")
scaler_lc = load_scaler("./Output_Model/scaler_lc.pkl")

# Daftar penyakit untuk inisialisasi retrievers
disease_list = ["HT", "DM", "KP"]

# Fungsi untuk memuat dokumen berbasis penyakit dalam format .txt
def load_documents_by_disease(disease):
    txt_folder_path = f"./Data/{disease}"
    all_txt_paths = glob.glob(os.path.join(txt_folder_path, "*.txt"))

    documents = []
    for txt_path in all_txt_paths:
        txt_path = txt_path.replace(os.sep, '/')
        loader = TextLoader(txt_path)
        txt_docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        documents.extend(text_splitter.split_documents(txt_docs))

    return documents

# Inisialisasi model LLM dan retrievers untuk semua penyakit
@st.cache_resource
def init_recommendation():
    # Load API key dan model
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)

    # Membuat retriever untuk setiap penyakit
    retrievers = {}
    for disease in disease_list:
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
    1. Obat yang disarankan beserta dosisnya.
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

# Fungsi untuk menampilkan input
def triple_column_input(inputs):
    col1, col2, col3 = st.columns(3)
    for i, (key, value) in enumerate(inputs.items()):
        col = [col1, col2, col3][i % 3]
        with col:
            if value['type'] == 'slider':
                st.slider(label=value['label'], min_value=value['min_value'], max_value=value['max_value'], step=value['step'], key=key)
            elif value['type'] == 'selectbox':
                st.selectbox(label=value['label'], options=value['options'], format_func=value['format_func'], key=key)
            elif value['type'] == 'number_input':
                st.number_input(label=value['label'], min_value=value['min_value'], max_value=value['max_value'], step=value['step'], key=key)

# Fungsi prediksi HT
def predict_ht():
    st.markdown("<h1 style='text-align: center;'>Prediksi Hipertensi</h1>", unsafe_allow_html=True)
    inputs = {
        "cp": {"label": "Tipe Sakit Data", "options": [0, 1, 2, 3], "type": "selectbox", 
               "format_func": lambda x: {0: "Asymptomatic", 1: "Typical Angina", 2: "Atypical Angina", 3: "Non-Anginal"}[x]},
        "trestbps": {"label": "Trestbps", "min_value": 50.0, "max_value": 200.0, "step": 0.1, "type": "number_input"},
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
        restecg = st.session_state.restecg
        thalach = st.session_state.thalach
        exang = st.session_state.exang
        oldpeak = st.session_state.oldpeak
        slope = st.session_state.slope
        ca = st.session_state.ca
        thal = st.session_state.thal
        
        # Membuat DataFrame untuk input
        input_data = pd.DataFrame([[cp, trestbps, restecg, thalach, exang, oldpeak, slope, ca, thal]], 
                                  columns=['cp', 'trestbps', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
        
        # Lakukan scaling pada data
        input_data_scaled = scaler_ht.transform(input_data)
        
        # Prediksi menggunakan model (model harus sudah terdefinisi)
        predicted_label = model_ht.predict(input_data_scaled)[0]
        proba = model_ht.predict_proba(input_data_scaled)[0]

        st.write("Probabilitas Negatif:", proba[0])
        st.write("Probabilitas Positif:", proba[1])
        hasil_prediksi = "Negatif" if predicted_label == 0 else "Positive"
        st.write(f"### Hasil Prediksi: {hasil_prediksi}")
        
        # Menyimpan hasil prediksi di session_state untuk halaman rekomendasi
        st.session_state['ht_prediction'] = {
            "cp": cp,
            "trestbps": trestbps,
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
    st.markdown("<h1 style='text-align: center;'>Prediksi Diabetes</h1>", unsafe_allow_html=True)
    inputs = {
        "Pregnancies": {"label": "Jumlah Kehamilan", "min_value": 0, "max_value": 20, "step": 1, "type": "number_input"},
        "Glucose": {"label": "Glukosa", "min_value": 0, "max_value": 200, "step": 1, "type": "number_input"},
        "BloodPressure": {"label": "Tekanan Darah Diastolik", "min_value": 0, "max_value": 150, "step": 1, "type": "number_input"},
        "SkinThickness": {"label": "Ketebalan Kulit Trisep", "min_value": 0, "max_value": 100, "step": 1, "type": "number_input"},
        "Insulin": {"label": "Insulin", "min_value": 0, "max_value": 1000, "step": 1, "type": "number_input"},
        "BMI": {"label": "BMI", "min_value": 10.0, "max_value": 100.0, "step": 0.1, "type": "number_input"},
        "DiabetesPedigreeFunction": {"label": "Peluang Diabetes-Riwayat Keluarga", "min_value": 0.050, "max_value": 2.500, "step": 0.001, "type": "number_input"},
        "Age": {"label": "Umur", "min_value": 0, "max_value": 100, "step": 1, "type": "number_input"}
    }
    triple_column_input(inputs)

    if st.button("Prediksi"):
        # Menangkap input pengguna
        Pregnancies = st.session_state['Pregnancies']
        Glucose = st.session_state['Glucose']
        BloodPressure = st.session_state['BloodPressure']
        SkinThickness = st.session_state['SkinThickness']
        Insulin = st.session_state['Insulin']
        BMI = st.session_state['BMI']
        DiabetesPedigreeFunction = st.session_state['DiabetesPedigreeFunction']
        Age = st.session_state['Age']
        
        # Membuat DataFrame untuk input
        input_data = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]], 
                                  columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
        
        # Lakukan scaling pada data
        # st.write(input_data)
        input_data_scaled = scaler_dm.transform(input_data)
        # st.write(input_data_scaled)
        # Prediksi menggunakan model (model harus sudah terdefinisi)
        predicted_label = model_dm.predict(input_data_scaled)[0]
        # Prediksi probabilitas
        proba = model_dm.predict_proba(input_data_scaled)[0]
        st.write("Probabilitas Negatif:", proba[0])
        st.write("Probabilitas Positif:", proba[1])
        
        # Prediksi label
        #predicted_label = model_dm.predict(input_data_scaled)[0]
        hasil_prediksi = "Negatif" if predicted_label == 0 else "Positive"
        st.write(f"### Hasil Prediksi: {hasil_prediksi}")
        
        # Menyimpan hasil prediksi di session_state untuk halaman rekomendasi
        st.session_state['dm_prediction'] = {
            "Pregnancies": Pregnancies,
            "Glucose": Glucose,
            "BloodPressure": BloodPressure,
            "SkinThickness": SkinThickness,
            "Insulin": Insulin,
            "BMI": BMI,
            "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
            "Age": Age,
            "hasil_prediksi": hasil_prediksi
        }
        
        # Arahkan pengguna ke halaman rekomendasi
        if st.button("Dapatkan Rekomendasi"):
            st.session_state.page = 'Recommendation'
            st.rerun()

def predict_lungcancer():
    st.markdown("<h1 style='text-align: center;'>Prediksi Kanker Paru-Paru</h1>", unsafe_allow_html=True)
    inputs = {
        "GENDER": {"label": "Jenis Kelamin", "options": [0, 1], "type": "selectbox", 
                "format_func": lambda x: "Wanita" if x == 0 else "Pria"},
        "AGE": {"label": "Umur", "min_value": 0, "max_value": 100, "step": 1, "type": "number_input"},
        "SMOKING": {"label": "Merokok", "options": [1, 2], "type": "selectbox", 
                "format_func": lambda x: "Tidak" if x == 1 else "Ya"},
        "YELLOW_FINGERS": {"label": "Jari Kuning", "options": [1, 2], "type": "selectbox", 
                "format_func": lambda x: "Tidak" if x == 1 else "Ya"},
        "ANXIETY": {"label": "Mengalami Kecemasan", "options": [1, 2], "type": "selectbox", 
                "format_func": lambda x: "Tidak" if x == 1 else "Ya"},
        "PEER_PRESSURE": {"label": "Tekanan Teman", "options": [1, 2], "type": "selectbox", 
                "format_func": lambda x: "Tidak" if x == 1 else "Ya"},
        "CHRONIC_DISEASE": {"label": "Memiliki Penyakit Kronis?", "options": [1, 2], "type": "selectbox", 
                "format_func": lambda x: "Tidak" if x == 1 else "Ya"},
        "FATIGUE": {"label": "Mengalami Kelelahan?", "options": [1, 2], "type": "selectbox", 
                "format_func": lambda x: "Tidak" if x == 1 else "Ya"},
        "ALLERGY": {"label": "Memiliki Alergu?", "options": [1, 2], "type": "selectbox", 
                "format_func": lambda x: "Tidak" if x == 1 else "Ya"},
        "WHEEZING": {"label": "Mengalami Mengi?", "options": [1, 2], "type": "selectbox", 
                "format_func": lambda x: "Tidak" if x == 1 else "Ya"},      
        "ALCOHOL_CONSUMING": {"label": "Konsumsi Alkohol?", "options": [1, 2], "type": "selectbox", 
                "format_func": lambda x: "Tidak" if x == 1 else "Ya"},
        "COUGHING": {"label": "Mengalami Batuk?", "options": [1, 2], "type": "selectbox", 
                "format_func": lambda x: "Tidak" if x == 1 else "Ya"},
        "SHORTNESS_OF_BREATH": {"label": "Mengalami Sesak Napas?", "options": [1, 2], "type": "selectbox", 
                "format_func": lambda x: "Tidak" if x == 1 else "Ya"},
        "SWALLOWING_DIFFICULTY": {"label": "Mengalami Kesulitan Menelan?", "options": [1, 2], "type": "selectbox", 
                "format_func": lambda x: "Tidak" if x == 1 else "Ya"},
        "CHEST_PAIN": {"label": "Mengalami Nyeri Dada?", "options": [1, 2], "type": "selectbox", 
                "format_func": lambda x: "Tidak" if x == 1 else "Ya"}
    }
    triple_column_input(inputs)

    if st.button("Prediksi"):
        # Menangkap input pengguna dan memastikan nama variabel sesuai dengan model pelatihan
        GENDER = st.session_state.GENDER
        AGE = st.session_state.AGE
        SMOKING = st.session_state.SMOKING
        YELLOW_FINGERS = st.session_state.YELLOW_FINGERS
        ANXIETY = st.session_state.ANXIETY
        PEER_PRESSURE = st.session_state.PEER_PRESSURE
        CHRONIC_DISEASE = st.session_state.CHRONIC_DISEASE
        FATIGUE = st.session_state.FATIGUE
        ALLERGY = st.session_state.ALLERGY
        WHEEZING = st.session_state.WHEEZING
        ALCOHOL_CONSUMING = st.session_state.ALCOHOL_CONSUMING
        COUGHING = st.session_state.COUGHING
        SHORTNESS_OF_BREATH = st.session_state.SHORTNESS_OF_BREATH
        SWALLOWING_DIFFICULTY = st.session_state.SWALLOWING_DIFFICULTY
        CHEST_PAIN = st.session_state.CHEST_PAIN
        
        # Membuat DataFrame untuk input
        input_data = pd.DataFrame([[GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, 
                                    CHRONIC_DISEASE, FATIGUE, ALLERGY, WHEEZING, ALCOHOL_CONSUMING, 
                                    COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN]], 
                                  columns=['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
                                           'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING',
                                           'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH',
                                           'SWALLOWING DIFFICULTY', 'CHEST PAIN'])

        # Lakukan scaling pada data menggunakan scaler yang telah terfit
        input_data_scaled = scaler_lc.transform(input_data)

        # Prediksi menggunakan model (model harus sudah terdefinisi)
        predicted_label = model_lc.predict(input_data_scaled)[0]
        proba = model_lc.predict_proba(input_data_scaled)[0]

        st.write("Probabilitas Negatif:", proba[0])
        st.write("Probabilitas Positif:", proba[1])
        hasil_prediksi = "Negatif" if predicted_label == 0 else "Positif"
        st.write(f"### Hasil Prediksi: {hasil_prediksi}")
        
        # Menyimpan hasil prediksi di session_state untuk halaman rekomendasi
        st.session_state['lc_prediction'] = {
            "GENDER": GENDER,
            "AGE": AGE,
            "SMOKING": SMOKING,
            "YELLOW_FINGERS": YELLOW_FINGERS,
            "ANXIETY": ANXIETY,
            "PEER_PRESSURE": PEER_PRESSURE,
            "CHRONIC_DISEASE": CHRONIC_DISEASE,
            "ALCOHOL_CONSUMING": ALCOHOL_CONSUMING,
            "COUGHING": COUGHING,
            "SWALLOWING_DIFFICULTY": SWALLOWING_DIFFICULTY,
            "CHEST_PAIN": CHEST_PAIN,
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
        'lc_prediction': 'Kanker Paru-Paru'
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
            st.markdown(f"**Rekomendasi {recommendation_type}:** {answer.content}")
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
    elif selected == 'Lung Cancer Prediction':
        predict_lungcancer()
    elif selected == 'Recommendation':
        show_recommendation()
    elif selected == 'About Us':
        about_us_section()

if __name__ == "__main__":
    main()