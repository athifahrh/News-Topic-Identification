import re
import spacy
import nltk
import subprocess
import pickle
import streamlit as st
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import models, corpora

st.set_page_config(
    page_title="Identifikasi Topik Berita",
    layout="wide"
)

nltk.download("stopwords")
nltk.download("punkt")

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

with st.sidebar:
    st.write("Pilih model")
    model_options = st.selectbox(
        "Label",
        options=["LDA", "Guided LDA"],
        index=None,
        placeholder="Klik di sini untuk memilih model",
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("""
        <div style='text-align: center;'>
            <p>Developed with ❤️ by<br>'Athifah Radhiyah H. (24050120140167)</p>
        </div>
        
    """, unsafe_allow_html=True)

    st.markdown("""
        <style>
            .social-icons {
                text-align: center;
            }
            .social-icons a {
                color: #FAFAFA;
                font-size: 30px;
                margin: 0 5px;
            }
        </style>
        <div class="social-icons">
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.5.0/font/bootstrap-icons.css" />
            <a href='https://www.linkedin.com/in/athifahrh/' target='_blank'><i class="bi bi-linkedin"></i></a>
            <a href='https://github.com/athifahrh' target='_blank'><i class="bi bi-github"></i></a>
            <a href='https://www.instagram.com/athifahrh/' target='_blank'><i class="bi bi-instagram"></i></a>
        </div>
    """, unsafe_allow_html=True)

st.html("""
    <style>
        [alt=Logo] {
            height: 3rem;
            border-radius: 10px;
            background-color: #FAFAFA;
            padding: 5px;
        }
    </style>
""")
st.logo(image="S1-Statistika.png", link="https://stat.fsm.undip.ac.id/v1/", icon_image=None)

st.markdown("""
    <h1 style='text-align: center;'>
        Identifikasi Topik Berita
    </h1>
""", unsafe_allow_html=True)

st.write("")
st.write("")

st.write("Silakan tulis berita di sini")
content = st.text_area(
    "label",
    placeholder="Tuliskan berita dalam bahasa Inggris",
    label_visibility="collapsed"
)

col1, col2 = st.columns([6, 1])
with col2:
    predict = st.button(
        "Prediksi",
        type="primary",
        use_container_width=True
    )

def cleaning(text):
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = " ".join(text.split())
    return text

def case_folding(text):
    text = text.lower()
    return text

def lemmatization(text, allowed_postags=["NOUN", "ADJ", "VERB", "ADV", "PROPN"]):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    doc = nlp(text)
    new_text = [token.lemma_ for token in doc if token.pos_ in allowed_postags]
    return " ".join(new_text)

if predict:
    if model_options and content:
        content = cleaning(content)
        content = case_folding(content)
        content = lemmatization(content)

        dictionary = corpora.Dictionary.load("models/dictionary.gensim")

        content_tokens = word_tokenize(content)
        content_tokens = [word for word in content_tokens if word not in stopwords.words("english") and len(word) > 2]
        content_bow = dictionary.doc2bow(content_tokens)

        topic_mapping = {
            0: "Konflik Israel",
            1: "Korupsi",
            2: "Isu Kriminal",
            3: "Pemilu",
            4: "Isu Pemerintahan"
        }

        if model_options == "LDA":
            lda_model = models.LdaModel.load("models/lda_model.gensim")

            topic_distribution = lda_model.get_document_topics(content_bow)

            if len(topic_distribution) > 1:
                max_prob_topic = max(topic_distribution, key=lambda x: x[1])
                predicted_topic = max_prob_topic[0]
                confidence = max_prob_topic[1]
            elif len(topic_distribution) == 1:
                predicted_topic, confidence = topic_distribution[0]

            predicted_topic = topic_mapping[predicted_topic]

            st.markdown(f"""
                <div style='text-align: center;'>
                    <h3 style='font-weight: normal;'>
                        Prediksi topik untuk berita di atas adalah <b>{predicted_topic}</b> dengan tingkat keyakinan <b>{confidence * 100:.2f}%</b>.
                    </h3>
                </div>
            """, unsafe_allow_html=True)

        else:
            with open("models/guidedlda_model.pickle", "rb") as file_handle:
                guidedldamodel = pickle.load(file_handle)
            
            topic_distribution = guidedldamodel.transform(np.array(content_bow))
            topic_distribution = topic_distribution.mean(axis=0)
            predicted_topic = np.argmax(topic_distribution)
            confidence = topic_distribution[predicted_topic]
            predicted_topic = topic_mapping[predicted_topic]

            st.markdown(f"""
                <div style='text-align: center;'>
                    <h3 style='font-weight: normal;'>
                        Prediksi topik untuk berita di atas adalah <b>{predicted_topic}</b> dengan tingkat keyakinan <b>{confidence * 100:.2f}%</b>.
                    </h3>
                </div>
            """, unsafe_allow_html=True)

    elif model_options and not content:
        st.error("Tuliskan berita yang ingin diprediksi terlebih dahulu.")
    elif not model_options and content:
        st.error("Pilih model terlebih dahulu.")
    else:
        st.error("Tuliskan berita yang ingin diprediksi dan pilih model terlebih dahulu.")