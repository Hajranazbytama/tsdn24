import streamlit as st
import plotly.graph_objects as go
import numpy as np

def home_section():
    # Judul
    st.markdown(
        """
        <h1 style="text-align: center; margin-bottom: 20px;">Selamat Datang 🙌😊<br> di Sistem Pemantauan PTM</h1>
        """, unsafe_allow_html=True
    )
    st.write("Sistem ini dapat membantu Anda dalam memantau kesehatan Anda pada Penyakit Tidak Menular (PTM)")

    # Menampilkan gambar penyakit PTM berjajar dengan style bulat
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div style="text-align: center;">
                <img src="https://img.freepik.com/free-vector/flat-world-hypertension-day-illustration_23-2148896416.jpg?t=st=1731311231~exp=1731314831~hmac=736c93a326b4e2f747a6a56da3cc212cedcc535081f6df5cdc3af7e17f52e86a&w=740" 
                     style="width: 150px; height: 150px; border-radius: 50%; margin-bottom: 10px;"/>
                <div style="font-weight: bold;">Hipertensi</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(
            """
            <div style="text-align: center;">
                <img src="https://img.freepik.com/free-vector/diabetes-flat-composition-medical-with-patient-symptoms-complications-blood-sugar-meter-treatments-medication_1284-28998.jpg?t=st=1731047428~exp=1731051028~hmac=6f72be42ef12428bc36df67f0e63bdd9583450f7d315343f7231a35bd8261467&w=826" 
                     style="width: 150px; height: 150px; border-radius: 50%; margin-bottom: 10px;"/>
                <div style="font-weight: bold;">Diabetes</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(
            """
            <div style="text-align: center;">
                <img src="https://img.freepik.com/free-vector/hand-drawn-cancer-infographic-template_23-2149225784.jpg?t=st=1731490910~exp=1731494510~hmac=d6fc2aea086765dbf3aa75c090c3fa73b82417682252d50ddc7fcb0df7db34e2&w=740" 
                     style="width: 150px; height: 150px; border-radius: 50%; margin-bottom: 10px;"/>
                <div style="font-weight: bold;">Kanker Paru-Paru</div>
            </div>
            """, unsafe_allow_html=True)

    # Menambahkan fun fact menarik tentang PTM
    st.markdown("""
    <div style="text-align: center; margin-top: 20px;">
        <h3 style="font-weight: bold;">Fun Facts 💭</h3>
        <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
            <div style="display: inline-block; width: 200px; padding: 10px; border: 1px solid #ddd; border-radius: 8px; background-color: #f9f9f9;">
               Hipertensi disebut  <strong>pembunuh senyap</strong> karena komplikasi seriusnya yang bisa muncul mendadak meskipun seringkali <strong>tidak bergejala</strong>.
            </div>
            <div style="display: inline-block; width: 200px; padding: 10px; border: 1px solid #ddd; border-radius: 8px; background-color: #f9f9f9;">
                <strong>Diabetes</strong> tidak hanya mempengaruhi gula darah, tapi juga bisa merusak organ-organ penting seperti <strong>ginjal, saraf, dan jantung</strong>.
            </div>
            <div style="display: inline-block; width: 200px; padding: 10px; border: 1px solid #ddd; border-radius: 8px; background-color: #f9f9f9;">
                Meskipun <strong>merokok</strong> adalah penyebab utama, kanker paru-paru juga bisa menyerang <strong>non-perokok</strong> akibat paparan asap rokok bekas, polusi, dan faktor genetik.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    def show_interactive_chart():
        # Data dummy untuk grafik kematian
        tahun = np.arange(2010, 2021)
        kematian_ppok = np.array([1000, 1200, 1500, 1800, 2000, 2300, 2600, 2900, 3200, 3500, 3800])
        kematian_dm = np.array([900, 1100, 1400, 1600, 1800, 2100, 2400, 2700, 3000, 3300, 3600])
        kematian_heart = np.array([800, 1000, 1300, 1500, 1700, 2000, 2300, 2600, 2900, 3200, 3500])

        # Membuat grafik interaktif dengan Plotly
        fig = go.Figure()

        # Menambahkan data PPOK
        fig.add_trace(go.Scatter(
            x=tahun, y=kematian_ppok, mode='lines+markers', name='PPOK',
            marker=dict(size=8), line=dict(width=2)))

        # Menambahkan data Diabetes Melitus
        fig.add_trace(go.Scatter(
            x=tahun, y=kematian_dm, mode='lines+markers', name='Diabetes Melitus',
            marker=dict(size=8), line=dict(width=2)))

        # Menambahkan data Penyakit Jantung
        fig.add_trace(go.Scatter(
            x=tahun, y=kematian_heart, mode='lines+markers', name='Penyakit Jantung',
            marker=dict(size=8), line=dict(width=2)))

        # Menambahkan label dan judul rata tengah
        fig.update_layout(
            xaxis_title="Tahun",
            yaxis_title="Jumlah Kematian",
            legend_title="Jenis Penyakit",
            hovermode="x unified"  # Menampilkan semua data dalam satu tooltip
        )

        # Menampilkan grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGrey")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGrey")

        # Menampilkan judul menggunakan Streamlit dengan sedikit margin bawah
        st.markdown(
            """
            <div style="text-align: center; font-size: 24px; font-weight: bold; margin-top: 20px; margin-bottom: -20px;">
                Kasus PTM di Indonesia 📈
            </div>
            """, 
            unsafe_allow_html=True
        )

        # Menampilkan grafik di Streamlit
        st.plotly_chart(fig, use_container_width=True)

        # Menambahkan teks sumber di bawah grafik
        st.markdown(
            """
            <div style="text-align: center; font-size: 12px; color: grey; margin-top: -10px;">
                Sumber: <a href="https://www.example.com" target="_blank">www.example.com</a>
            </div>
            """, 
            unsafe_allow_html=True
        )

    # Panggil fungsi untuk menampilkan grafik interaktif
    show_interactive_chart()