# The app begins here

import streamlit as st
from st_audiorec import st_audiorec
import soundfile as sf
from datetime import timedelta
import pandas as pd

from cosine_similarity import CosineSimilarityEngine


@st.cache_data(ttl=timedelta(days=1))
def setup_engine():
    cosine_similarity_engine = CosineSimilarityEngine()
    cosine_similarity_engine.train_model()
    return cosine_similarity_engine

st.set_page_config(
    page_title="Smart Recruitment Engine",
    layout = "wide"
)

st.title("""Smart Recruitment Engine""")

with st.expander("Help"):
    st.write("""Help meeeeeee""")
    st.write("LOREM IPSUM")

engine = setup_engine()
submitted = False

with st.form("SRE Form"):

    st.header("Candidate Assessment")

    input_candidate_id = st.text_input(label="Candidate ID")

    wav_audio_data = st_audiorec()

    if wav_audio_data is not None:
        st.audio(wav_audio_data, format='audio/wav')

    submitted = st.form_submit_button("Submit")


if submitted :
    
    roles_report = engine.result(['Kubernetes'])

    with st.container():
        st.header("Suitable Roles")
        st.subheader("The suggested roles for the candidate with ID "+ str(input_candidate_id)+" in descending order of match score are: ")
        roles_report_pd = pd.DataFrame(data=roles_report, columns=['Roles','Similarity Score'])
        st.dataframe(roles_report, use_container_width = True)
        #st.write(roles_report)