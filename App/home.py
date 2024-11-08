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
                <img src="https://img.freepik.com/free-vector/hand-drawn-flat-copd-illustration_23-2149101819.jpg?t=st=1731041425~exp=1731045025~hmac=28548e56f9f3ee68fee82388c9a3035669beab1f5a10f012b010fdd0d2b849df&w=826" 
                     style="width: 150px; height: 150px; border-radius: 50%; margin-bottom: 10px;"/>
                <div style="font-weight: bold;">PPOK</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(
            """
            <div style="text-align: center;">
                <img src="https://img.freepik.com/free-vector/diabetes-flat-composition-medical-with-patient-symptoms-complications-blood-sugar-meter-treatments-medication_1284-28998.jpg?t=st=1731047428~exp=1731051028~hmac=6f72be42ef12428bc36df67f0e63bdd9583450f7d315343f7231a35bd8261467&w=826" 
                     style="width: 150px; height: 150px; border-radius: 50%; margin-bottom: 10px;"/>
                <div style="font-weight: bold;">Diabetes Melitus</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(
            """
            <div style="text-align: center;">
                <img src="https://img.freepik.com/free-photo/cartoon-anatomical-heart-with-band-aids_23-2149767960.jpg?t=st=1731047336~exp=1731050936~hmac=95550ab4faa09b26a437b6b93ab810892cebf94f77be08836a1d84d7c6fdc460&w=826" 
                     style="width: 150px; height: 150px; border-radius: 50%; margin-bottom: 10px;"/>
                <div style="font-weight: bold;">Penyakit Jantung</div>
            </div>
            """, unsafe_allow_html=True)

    # Menambahkan fun fact menarik tentang PTM
    st.markdown("""
    <div style="text-align: center; margin-top: 20px;">
        <h3 style="font-weight: bold;">Fun Facts 💭</h3>
        <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
            <div style="display: inline-block; width: 200px; padding: 10px; border: 1px solid #ddd; border-radius: 8px; background-color: #f9f9f9;">
                <strong>Penyakit Tidak Menular (PTM)</strong> seperti PPOK, Diabetes, dan Penyakit Jantung merupakan penyebab utama kematian di seluruh dunia.
            </div>
            <div style="display: inline-block; width: 200px; padding: 10px; border: 1px solid #ddd; border-radius: 8px; background-color: #f9f9f9;">
                PTM sering kali disebabkan oleh faktor gaya hidup seperti pola makan yang buruk, kurangnya aktivitas fisik, dan merokok.
            </div>
            <div style="display: inline-block; width: 200px; padding: 10px; border: 1px solid #ddd; border-radius: 8px; background-color: #f9f9f9;">
                <strong>Diabetes Melitus</strong> dapat menyebabkan komplikasi serius seperti kerusakan ginjal, kebutaan, dan serangan jantung.
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