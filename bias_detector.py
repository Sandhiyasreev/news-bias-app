from transformers import pipeline

zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

def detect_bias(text):
    labels = ["left", "center", "right"]
    result = zero_shot(text, candidate_labels=labels)
    return dict(zip(result['labels'], result['scores']))

def detect_emotion(text):
    result = emotion_classifier(text)[0]
    return {r['label']: r['score'] for r in result}
