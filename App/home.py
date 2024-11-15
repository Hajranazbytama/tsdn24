import streamlit as st
import plotly.graph_objects as go
import numpy as np

def home_section():
    # Judul
    st.markdown(
        """
        <h1 style="text-align: center; margin-bottom: 20px;">Selamat Datang 🙌😊</h1>
        <h3 style="text-align: center; margin-bottom: 20px;">PTM-PRe: Aplikasi Prediksi Penyakit Tidak Menular dan Rekomendasi Personal Berbasis AI</h3>
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

    st.markdown(
        """
        <h3 style="text-align: center; margin-top: 40px;, font-weight:bold;">Perkembangan Prevalensi PTM di Indonesia</h3>
        """, unsafe_allow_html=True
    )

    def show_interactive_chart():
        # Data prevalensi penyakit yang ada di 2013, 2018, dan SKI 2023
        tahun = ['2013', '2018', '2023']
        
        # Prevalensi penyakit untuk tahun 2013
        prevalensi_2013 = {
            'Asma': 4.5, 'Kanker': 3.7, 'Diabetes Melitus': 1.5,
            'Hipertensi': 9.4, 'Stroke': 7.0, 'Penyakit Jantung': 0.5
        }
        
        # Prevalensi penyakit untuk tahun 2018
        prevalensi_2018 = {
            'Asma': 2.4, 'Kanker': 1.79, 'Diabetes Melitus': 1.5,
            'Hipertensi': 8.36, 'Stroke': 10.9, 'Penyakit Jantung': 1.5
        }
        
        # Prevalensi penyakit untuk SKI 2023
        prevalensi_ski_2023 = {
            'Asma': 1.6, 'Kanker': 1.2, 'Diabetes Melitus': 1.7,
            'Hipertensi': 8.0, 'Stroke': 8.3, 'Penyakit Jantung': 0.85
        }

        # Penyakit yang ada di ketiga tahun
        penyakit = list(prevalensi_2013.keys())  # Penyakit yang ada di ketiga tahun
        
        # Menyusun data untuk plot
        fig = go.Figure()

        # Menambahkan data prevalensi setiap penyakit
        for p in penyakit:
            # Mengambil nilai prevalensi untuk setiap tahun
            prevalensi_2013_vals = prevalensi_2013.get(p, 0)
            prevalensi_2018_vals = prevalensi_2018.get(p, 0)
            prevalensi_ski_2023_vals = prevalensi_ski_2023.get(p, 0)
            
            # Menambahkan garis untuk penyakit tersebut
            fig.add_trace(go.Scatter(
                x=tahun, y=[prevalensi_2013_vals, prevalensi_2018_vals, prevalensi_ski_2023_vals], 
                mode='lines+markers', name=p,
                marker=dict(size=8), line=dict(width=2)
            ))

        # Menambahkan label dan judul rata tengah
        fig.update_layout(
            xaxis_title="Tahun",
            yaxis_title="Prevalensi (%)",
            legend_title="Nama Penyakit",
            hovermode="x unified"
        )

        # Menampilkan grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGrey")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGrey")

        # Menampilkan grafik di Streamlit
        st.plotly_chart(fig, use_container_width=True)

        # Menambahkan teks sumber di bawah grafik
        st.markdown(
            """
            <div style="text-align: center; font-size: 12px; color: grey; margin-top: -10px;">
                Sumber: Riset Kesehatan Dasar (Riskesdas) 2013, 2018, dan SKI 2023
            </div>
            """, 
            unsafe_allow_html=True
        )

    # Panggil fungsi untuk menampilkan grafik interaktif
    show_interactive_chart()