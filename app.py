import streamlit as st
from transformers import pipeline
from utils import read_pdf, read_txt
import plotly.graph_objects as go
from newspaper import Article
from youtube_transcript_api import YouTubeTranscriptApi
from fpdf import FPDF
import re

# Load models with caching to avoid reloading every time
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    bias_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
    sentiment_classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    return summarizer, bias_classifier, emotion_classifier, sentiment_classifier

def extract_text_from_url(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text

def extract_text_from_youtube(video_url):
    try:
        match = re.search(r"(?:v=|be/|embed/)([a-zA-Z0-9_-]{11})", video_url)
        if not match:
            return "[Error] Invalid YouTube URL format"
        video_id = match.group(1)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry['text'] for entry in transcript])
    except Exception as e:
        return f"[Error] Couldn't extract transcript: {e}"

def plot_emotion_radar(emotion_scores):
    labels = list(emotion_scores.keys())
    values = list(emotion_scores.values())
    values += values[:1]
    labels += labels[:1]
    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=values,
                theta=labels,
                fill='toself',
                line=dict(color='blue'),
                name='Emotions'
            )
        ],
        layout=go.Layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False
        )
    )
    st.plotly_chart(fig)

def generate_report(summary, bias_scores, emotion_scores, sentiment):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="News Analysis Report", ln=True, align='C')
    pdf.ln(10)
    pdf.multi_cell(0, 10, f"Summary:\n{summary}")
    pdf.ln(5)

    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(200, 10, "Political Bias Scores:", ln=True)
    pdf.set_font("Arial", size=12)
    for k, v in bias_scores.items():
        pdf.cell(200, 10, f"{k.title()}: {round(v, 2)}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(200, 10, "Emotion Scores:", ln=True)
    pdf.set_font("Arial", size=12)
    for k, v in emotion_scores.items():
        pdf.cell(200, 10, f"{k.title()}: {round(v, 2)}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(200, 10, f"Sentiment: {sentiment}", ln=True)

    file_path = "report.pdf"
    pdf.output(file_path)
    return file_path

summarizer, bias_classifier, emotion_classifier, sentiment_classifier = load_models()

st.set_page_config(page_title="News Bias & Summary App", layout="wide")
st.title("📰 News Article Summarization & Bias Detection")

input_method = st.radio("Choose input method:", ("Paste Text", "Upload File", "Paste YouTube URL", "Paste Blog/Article URL"))

text = ""

if input_method == "Paste Text":
    text = st.text_area("Paste your news article below:")

elif input_method == "Upload File":
    uploaded_file = st.file_uploader("Upload .txt or .pdf", type=["txt", "pdf"])
    if uploaded_file:
        if uploaded_file.name.endswith(".pdf"):
            text = read_pdf(uploaded_file)
        elif uploaded_file.name.endswith(".txt"):
            text = read_txt(uploaded_file)

elif input_method == "Paste YouTube URL":
    youtube_url = st.text_input("Paste the YouTube video URL:")
    if youtube_url:
        with st.spinner("Extracting transcript..."):
            text = extract_text_from_youtube(youtube_url)
            st.success("YouTube transcript loaded!")

elif input_method == "Paste Blog/Article URL":
    blog_url = st.text_input("Paste the blog or article URL:")
    if blog_url:
        with st.spinner("Extracting article..."):
            text = extract_text_from_url(blog_url)
            st.success("Article content loaded!")

if text:
    st.write("✏️ **Extracted Text Preview:**")
    st.code(text[:1000])

    st.subheader("✂️ Summarization")
    with st.spinner("Summarizing article..."):
        summary = summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
        st.success(summary)

    st.subheader("🧭 Political Bias Detection")
    with st.spinner("Detecting political bias..."):
        labels = ["left", "center", "right"]
        bias_result = bias_classifier(text, candidate_labels=labels)
        bias_scores = dict(zip(bias_result['labels'], bias_result['scores']))
        st.bar_chart(bias_scores)

    st.subheader("💬 Emotion Analysis")
    with st.spinner("Analyzing emotion..."):
        truncated_text = text[:512]
        emotion_result = emotion_classifier(truncated_text)[0]
        emotion_scores = {entry['label']: entry['score'] for entry in emotion_result}
        plot_emotion_radar(emotion_scores)

    st.subheader("❤️ Sentiment Analysis")
    with st.spinner("Detecting sentiment..."):
        sentiment_result = sentiment_classifier(text[:512])[0]
        label_map = {
            "LABEL_0": "NEGATIVE",
            "LABEL_1": "NEUTRAL",
            "LABEL_2": "POSITIVE"
        }
        raw_label = sentiment_result['label']
        label = label_map.get(raw_label, raw_label)
        score = round(sentiment_result['score'], 2)
        emoji = {"POSITIVE": "😊", "NEGATIVE": "😠", "NEUTRAL": "😐"}
        st.markdown(f"**Sentiment:** {emoji.get(label, '')} {label} ({score})")
        st.progress(score)

    if st.button("📥 Download Report"):
        sentiment_text = f"{label} ({score})"
        path = generate_report(summary, bias_scores, emotion_scores, sentiment_text)
        with open(path, "rb") as f:
            st.download_button("Download PDF", f, file_name="News_Report.pdf")
