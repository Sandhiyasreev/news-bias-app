# 📰 News Article Summarization & Bias Detection App 🧠💬📊  
A powerful NLP application that summarizes articles, detects political bias, and analyzes emotions & sentiment using Transformer models.

---

# 🔧 Features  
✂️ Abstractive summarization using BART  
🧭 Political bias detection (Left, Center, Right) using zero-shot classification  
💬 Emotion detection: Happy, Angry, Sad, Hope, Fear, etc.  
❤️ Sentiment analysis: Positive, Negative, Neutral (with confidence score + emoji)  
📥 Input support:  
   - Paste text  
   - Upload .pdf or .txt files  
   - Paste YouTube/Blog URL (auto text extraction)  
📊 Emotion Radar Chart using Plotly  
📄 Downloadable PDF Report of summary + analysis  
☁️ Streamlit-powered web app interface  

---

# 🧠 Technologies Used  
- Python, Hugging Face Transformers  
- facebook/bart-large-cnn for summarization  
- bart-large-mnli for zero-shot bias detection  
- distilroberta for emotion detection  
- roberta-sentiment for sentiment analysis  
- newspaper3k & youtube-transcript-api for blog/video text extraction  
- FPDF for PDF report generation  
- Streamlit for UI  
- Plotly for emotion radar visualization  

---

# 🗂️ Project Structure  
news-bias-app/
├── app.py # Streamlit app code
├── utils.py # Helper functions (PDF, TXT readers)
├── requirements.txt # Python dependencies
├── README.md # This file
├── report.pdf # Sample output report
├── venv/ # (Should be ignored in .gitignore)
└── .gitignore # Ignore cache, venv, large files

---

# 🙋‍♀️ Created By  
Sandhiya Sree V
🔗 [LinkedIn](https://www.linkedin.com/in/sandhiya-sree-v-3a2321298/)
🌐 [GitHub](https://github.com/Sandhiyasreev)

# 📄 License 
This project is open source and available under the MIT License.
