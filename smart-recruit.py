# The app begins here

import streamlit as st
from st_audiorec import st_audiorec
import soundfile as sf
from datetime import timedelta
import pandas as pd

from cosine_similarity import CosineSimilarityEngine
from speech2text import Speech2TextEngine, TextSummarizationEngine
from question_generator import QuestionGeneratorEngine


@st.cache_resource(ttl=timedelta(days=1))
def setup_cossim_engine():
    engine = CosineSimilarityEngine()
    engine.train_model()
    return engine


@st.cache_resource(ttl=timedelta(days=1))
def setup_conversion_engine():
    engine = Speech2TextEngine()
    return engine


@st.cache_resource(ttl=timedelta(days=1))
def setup_summarization_engine():
    engine = TextSummarizationEngine()
    engine.load_model()
    return engine


@st.cache_resource(ttl=timedelta(days=1))
def setup_question_engine():
    engine = QuestionGeneratorEngine()
    engine.load_model()
    return engine


@st.cache_data(ttl=timedelta(days=1))
def transcript_summary():
    transcript = conversion_engine.conversion("harvard.wav")
    summary = summarization_engine.summarize(transcript)
    return summary


def question_generator():
    questions = question_engine.generate_questions(context)
    return questions


st.set_page_config(page_title="Smart Recruitment Engine", layout="wide", page_icon="⚡")

st.title("""⚡ Smart Recruitment Engine""")

with st.expander("Help"):
    st.write("""AI powered recruitments!
    

    I need help.

    Cool, all you have to do is,
    - Record the conversation between you (interviewer) and the candidate
    - Wait for the machine to generate adaptive questions
    - Once the interview is done, hit submit
    - Voila! You can now see what roles seem to be the best fit for the candidate 😊
    """)

cossim_engine = setup_cossim_engine()
conversion_engine = setup_conversion_engine()
summarization_engine = setup_summarization_engine()
question_engine = setup_question_engine()

submitted = False

with st.form("SRE Form"):
    st.header("Candidate Assessment")

    input_candidate_id = st.text_input(label="Candidate ID")

    wav_audio_data = st_audiorec()

    if wav_audio_data is not None:
        st.audio(wav_audio_data, format="audio/wav")

        context = transcript_summary()
        questions = question_generator()

        st.write(questions)

    submitted = st.form_submit_button("Submit")


if submitted:
    roles_report = cossim_engine.result(["Kubernetes"])

    with st.container():
        st.header("Suitable Roles")
        st.text(
            "The suggested roles for the candidate with ID "
            + str(input_candidate_id)
            + " in the order of best scores :"
        )
        roles_report_pd = pd.DataFrame(
            data=roles_report, columns=("Roles", "Similarity Score")
        )
        st.dataframe(
            roles_report,
            use_container_width=True,
            column_config={"": "Role", "value": "Score"},
        )
