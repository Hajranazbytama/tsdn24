import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense
import os
import glob
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema import HumanMessage

# KOMENTAR TEST

# Initialize session state for navigation
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# Sidebar navigation
st.sidebar.title('Navigation')

# Navigation buttons
if st.sidebar.button('HOME'):
    st.session_state.page = 'Home'
if st.sidebar.button('PREDICTION'):
    st.session_state.page = 'Prediction'
if st.sidebar.button('RECOMMENDATION'):
    st.session_state.page = 'Recommendation'
if st.sidebar.button('About Us'):
    st.session_state.page = 'About'

# Recommendation System Functions
@st.cache_resource
def init_recommendation():
    # Load all PDFs from the specified folder
    pdf_folder_path = "./Data/"
    all_pdf_paths = glob.glob(os.path.join(pdf_folder_path, "*.pdf"))
    
    documents = []
    for pdf_path in all_pdf_paths:
        loader = PyPDFLoader(pdf_path)
        pdf_docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        documents.extend(text_splitter.split_documents(pdf_docs))
    
    GEMINI_API_KEY = "AIzaSyDFQrUxPXyeVGU66oxymNMeK9IZy_Z272U"  # Replace with your actual API key
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)
    
    vector_db = FAISS.from_documents(documents, embeddings)
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    return llm, retriever

# Define RAG prompt templates for different recommendations
def generate_treatment_prompt(query, context):
    prompt = f"""
    Anda adalah seorang ahli kesehatan yang membantu petugas kesehatan untuk memberikan rekomendasi pengobatan kepada pasien berdasarkan informasi yang tersedia.

    **Profil dan Riwayat Pasien**:
    {query}

    **Riwayat Medis dan Keterangan Medis**:
    {context}

    Berdasarkan informasi di atas, berikan rekomendasi pengobatan yang singkat namun spesifik dan jelas meliputi:
    1. Obat yang disarankan beserta dosisnya (jika mungkin).
    2. Metode pengobatan yang sesuai.
    3. Langkah perawatan yang harus dilakukan oleh petugas medis terhadap pasien.
    """
    return prompt

def generate_lifestyle_prompt(query, context):
    prompt = f"""
    Anda adalah seorang ahli kesehatan yang membantu petugas kesehatan memberikan rekomendasi pola hidup sehat kepada pasien yang terindikasi TB atau tidak dengan informasi berikut.

    **Profil dan Riwayat Pasien**:
    {query}

    **Riwayat Medis dan Keterangan Medis**:
    {context}

    Berdasarkan informasi di atas, berikan rekomendasi pola hidup yang singkat namun spesifik dan jelas yang mencakup:
    1. Aktivitas fisik yang aman dan direkomendasikan (misalnya, jenis olahraga dan frekuensinya).
    2. Pola makan dan jenis makanan yang sebaiknya dikonsumsi dan dihindari (contoh: makanan yang meningkatkan imunitas).
    3. Kebiasaan sehari-hari yang dapat membantu pemulihan, termasuk tips manajemen stres dan tidur.
    4. Instruksi khusus untuk menjaga kebersihan dan mencegah penularan.
    """
    return prompt

def generate_followup_prompt(query, context):
    prompt = f"""
    Anda adalah seorang ahli kesehatan yang memberikan rekomendasi penanganan lanjutan bagi petugas kesehatan untuk pasien yang terindikasi TB atau tidak dengan informasi berikut.

    **Profil dan Riwayat Pasien**:
    {query}

    **Riwayat Medis dan Keterangan Medis**:
    {context}

    Berdasarkan informasi di atas, berikan rekomendasi penanganan lanjutan yang singkat namun spesifik dan jelas mencakup:
    1. Jadwal kontrol kesehatan atau pemeriksaan lanjutan yang disarankan.
    2. Pengujian tambahan atau pemeriksaan yang mungkin diperlukan (contoh: X-ray atau tes laboratorium).
    3. Tanda atau gejala yang perlu diwaspadai sebagai indikasi komplikasi.
    4. Saran untuk pemulihan yang berkelanjutan, seperti adaptasi pola hidup, manajemen stres, dan dukungan sosial yang dibutuhkan.
    """
    return prompt

# Prediction System Functions
@st.cache_resource
def init_prediction():
    # Create dummy data
    np.random.seed(42)
    data = pd.DataFrame({
        'age': np.random.randint(40, 80, 100),
        'smoking_history': np.random.choice([0, 1], size=100),
        'exposure_to_pollution': np.random.uniform(0, 100, 100),
        'lung_function': np.random.uniform(0.6, 1.2, 100)
    })
    data['risk'] = np.where(data['lung_function'] < 0.8, 1, 0)
    
    # Preprocessing
    scaler = MinMaxScaler()
    data[['age', 'smoking_history', 'exposure_to_pollution', 'lung_function']] = scaler.fit_transform(
        data[['age', 'smoking_history', 'exposure_to_pollution', 'lung_function']]
    )
    
    # Train model
    X = data[['age', 'smoking_history', 'exposure_to_pollution', 'lung_function']].values
    y = data['risk'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=4, validation_data=(X_test, y_test))
    
    return model, scaler

def predict_risk(model, scaler, input_data):
    input_data = scaler.transform(input_data)
    input_data = input_data.reshape((input_data.shape[0], 1, input_data.shape[1]))
    prediction = model.predict(input_data)
    return prediction

def get_recommendation(prediction):
    if prediction > 0.5:
        return "Risiko PPOK tinggi: Hindari paparan polusi, lakukan pemeriksaan fungsi paru secara berkala, pertimbangkan berhenti merokok."
    else:
        return "Risiko PPOK rendah: Pertahankan gaya hidup sehat dan hindari paparan polusi."

# Page Functions
def show_home():
    st.title("Selamat Datang di Sistem Pemantauan Kesehatan")
    st.write("""
    Sistem ini menyediakan dua layanan utama:
    1. Prediksi risiko PPOK berdasarkan data kesehatan Anda
    2. Rekomendasi kesehatan personal berdasarkan profil medis Anda
    
    Silakan pilih menu di sidebar untuk mengakses layanan yang Anda butuhkan.
    """)

def show_prediction():
    st.title("Prediksi Risiko PPOK")
    
    model, scaler = init_prediction()
    
    age = st.number_input("Masukkan Usia:", min_value=0, max_value=120)
    smoking_history = st.selectbox("Pernah Merokok? (1: merokok, 0: tidak merokok)", [0, 1])
    exposure_to_pollution = st.slider("Paparan Polusi (%)", 0.0, 100.0, 50.0)
    lung_function = st.slider("Fungsi Paru-paru", 0.6, 1.2, 0.8)
    
    input_data = np.array([[age, smoking_history, exposure_to_pollution, lung_function]])
    
    if st.button("Prediksi Risiko"):
        risk_prediction = predict_risk(model, scaler, input_data)
        risk_level = "Tinggi" if risk_prediction > 0.5 else "Rendah"
        recommendation = get_recommendation(risk_prediction)
        
        st.write(f"Prediksi Risiko PPOK: **{risk_level}**")
        st.write(f"Rekomendasi: {recommendation}")
        
        if risk_prediction > 0.5:
            st.warning("Risiko PPOK Tinggi! Segera lakukan tindakan preventif untuk menanganinya.")
        else:
            st.success("Risiko PPOK Rendah, tetap jaga kesehatan!")

def show_recommendation():
    st.title("Sistem Rekomendasi Kesehatan")
    
    llm, retriever = init_recommendation()
    
    profil_pasien = st.text_input("Masukkan Profil Pasien (umur, jenis kelamin, dll):")
    riwayat_pasien = st.text_area("Masukkan Riwayat Pasien:")
    pola_hidup = st.text_area("Masukkan Pola Hidup Pasien:")
    hasil_ctscan = st.selectbox("Masukkan Hasil CT Scan", ("TB", "Tidak TB"))
    
    recommendation_type = st.radio("Pilih Jenis Rekomendasi:",
                                 ("Rekomendasi Pengobatan",
                                  "Rekomendasi Pola Hidup",
                                  "Rekomendasi Penanganan Lanjutan"))
    
    if st.button("Dapatkan Rekomendasi"):
        query = f"Profil pasien: {profil_pasien}. Riwayat: {riwayat_pasien}. Pola Hidup:{pola_hidup}. Hasil CT Scan: {hasil_ctscan}."
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