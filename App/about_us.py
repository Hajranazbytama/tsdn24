# about_us.py
import streamlit as st

def about_us_section():
    st.markdown(
        """
        <h1 style="text-align: center; margin-bottom: 50px;">Web App Contributor</h1>
        """, unsafe_allow_html=True
    )

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