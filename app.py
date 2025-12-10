from flask import Flask, render_template_string, request, url_for
from werkzeug.utils import secure_filename
import os

# --- NEW: imports for ML + OCR ---
import io
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

from LughaatNLP import LughaatNLP
from google.cloud import vision

# =========================================================
# 0. Paths / Credentials
# =========================================================

# ❶ Path to your CSV (adjust if different)
DATA_PATH = r"D:\Umer Notes\7th SEM\ML\PROJECT\combined_articles.csv"

if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"D:\Umer Notes\7th SEM\ML\PROJECT\KEYS\urdu-ocr-480714-fc60bb78cd98.json"


app = Flask(__name__, static_url_path="/static", static_folder="static")

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# We'll fill CLASSES after loading dataset
CLASSES = []

# =========================================================
# 1. Load dataset + train Naive Bayes
# =========================================================

print("Loading dataset...")
dataset = pd.read_csv(DATA_PATH)
dataset = dataset[["content", "gold_label"]].dropna()

# -----------------------------
# Preprocessing with LughaatNLP
# -----------------------------
print("Initializing LughaatNLP...")
preprocessor = LughaatNLP()

def preprocess(text: str):
    text = preprocessor.normalize(text)
    text = preprocessor.remove_stopwords(text)
    text = preprocessor.lemmatize_sentence(text)
    tokens = preprocessor.urdu_tokenize(text)
    return tokens

def add_bigrams(tokens):
    bigrams = [tokens[i] + "_" + tokens[i+1] for i in range(len(tokens)-1)]
    return tokens + bigrams

print("Preprocessing dataset (this may take a bit)...")
dataset["content"] = dataset["content"].astype(str).apply(preprocess)
dataset["content"] = dataset["content"].apply(add_bigrams)

documents = dataset["content"]
labels = dataset["gold_label"]

CLASSES = sorted(dataset["gold_label"].unique())
print("Classes:", CLASSES)

# -----------------------------
# Naive Bayes training
# -----------------------------

def calculate_class_probabilities(y_train):
    class_probabilities = defaultdict(int)
    total = len(y_train)
    for label in y_train:
        class_probabilities[label] += 1
    for label in class_probabilities:
        class_probabilities[label] /= total
    return class_probabilities

def train_naive_bayes(X_train, y_train, min_freq=3, max_doc_ratio=0.8, bernoulli=False):
    global_counts = Counter()
    doc_counts = Counter()
    num_docs = len(X_train)

    # Global term stats
    for doc in X_train:
        tokens = list(doc)
        global_counts.update(tokens)
        doc_counts.update(set(tokens))

    # Build vocabulary
    vocab = set()
    for word, count in global_counts.items():
        if count < min_freq:
            continue
        if (doc_counts[word] / num_docs) > max_doc_ratio:
            continue
        vocab.add(word)

    # Class word frequencies
    class_word_freq = defaultdict(lambda: defaultdict(int))
    class_doc_count = defaultdict(int)

    for doc, label in zip(X_train, y_train):
        class_doc_count[label] += 1
        tokens = set(doc) if bernoulli else doc
        for word in tokens:
            if word in vocab:
                class_word_freq[label][word] += 1

    return vocab, class_word_freq, class_doc_count

print("Training Naive Bayes model on full dataset...")
vocab, class_word_freq, class_doc_count = train_naive_bayes(
    documents,
    labels,
    min_freq=3,
    max_doc_ratio=0.8,
    bernoulli=False
)

class_probabilities = calculate_class_probabilities(labels)

# Use your tuned alpha here if you have it
ALPHA = 0.5  # example

def predict_naive_bayes(doc_tokens):
    """
    Predict label for a single document (list of tokens).
    """
    vocab_size = len(vocab)
    scores = {}

    for label in class_probabilities:
        log_prob = np.log(class_probabilities[label])
        total_words = sum(class_word_freq[label].values())

        for word in doc_tokens:
            if word not in vocab:
                continue
            freq = class_word_freq[label].get(word, 0)
            log_prob += np.log((freq + ALPHA) / (total_words + ALPHA * vocab_size))

        scores[label] = log_prob

    return max(scores, key=scores.get)

def predict_headline(text: str) -> str:
    """
    Raw Urdu headline -> preprocess -> bigrams -> Naive Bayes -> label (string)
    """
    tokens = preprocess(text)
    tokens = add_bigrams(tokens)
    label = predict_naive_bayes(tokens)
    return label

print("Naive Bayes model ready.")

# =========================================================
# 2. OCR: Google Cloud Vision + NB
# =========================================================

def ocr_cloud_vision(image_path: str) -> str:
    """
    Uses Google Cloud Vision to extract text from an image.
    """
    client = vision.ImageAnnotatorClient()

    with io.open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)

    if response.error.message:
        raise Exception(f"Google Vision API Error: {response.error.message}")

    texts = response.text_annotations
    if not texts:
        return ""

    full_text = texts[0].description
    return full_text

def predict_from_image(image_path: str):
    """
    OCR (Google Vision) + preprocessing + Naive Bayes classification.
    Returns (ocr_text, predicted_label)
    """
    raw_text = ocr_cloud_vision(image_path)

    if not raw_text.strip():
        return "", None

    tokens = preprocess(raw_text)
    tokens = add_bigrams(tokens)
    label = predict_naive_bayes(tokens)
    return raw_text, label

# =========================================================
# 3. HTML Template (same as your UI)
# =========================================================

HTML_TEMPLATE = """
<!doctype html>
<html lang="ur" dir="rtl">
<head>
    <meta charset="utf-8">
    <title>Urdu News Headline Classifier</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">

    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    
    <!-- Animate.css -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css">

    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;600&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    
    <!-- Font Awesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">

    <style>
        :root {
            --primary: #6366f1;
            --primary-dark: #4f46e5;
            --secondary: #06b6d4;
            --accent: #f59e0b;
            --success: #10b981;
            --dark: #0f172a;
            --glass-bg: rgba(255, 255, 255, 0.95);
            --glass-border: rgba(255, 255, 255, 0.2);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: "Noto Nastaliq Urdu", "Inter", system-ui, sans-serif;
            min-height: 100vh;
            overflow-x: hidden;
            color: var(--dark);
        }

        /* Animated Gradient Background */
        .bg-animated {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(-45deg, #0f172a, #1e293b, #312e81, #1e1b4b);
            background-size: 400% 400%;
            animation: gradientShift 15s ease infinite;
            z-index: -2;
        }

        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* Floating Orbs */
        .orb {
            position: fixed;
            border-radius: 50%;
            filter: blur(80px);
            opacity: 0.5;
            z-index: -1;
            animation: float 20s ease-in-out infinite;
        }

        .orb-1 {
            width: 600px;
            height: 600px;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            top: -200px;
            right: -200px;
            animation-delay: 0s;
        }

        .orb-2 {
            width: 400px;
            height: 400px;
            background: linear-gradient(135deg, var(--secondary), var(--success));
            bottom: -100px;
            left: -100px;
            animation-delay: -5s;
        }

        .orb-3 {
            width: 300px;
            height: 300px;
            background: linear-gradient(135deg, var(--accent), var(--primary));
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            animation-delay: -10s;
        }

        @keyframes float {
            0%, 100% { transform: translate(0, 0) scale(1); }
            25% { transform: translate(30px, -30px) scale(1.05); }
            50% { transform: translate(-20px, 20px) scale(0.95); }
            75% { transform: translate(20px, 10px) scale(1.02); }
        }

        /* Particles */
        .particles {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            overflow: hidden;
        }

        .particle {
            position: absolute;
            width: 4px;
            height: 4px;
            background: rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            animation: rise 15s linear infinite;
        }

        @keyframes rise {
            0% {
                transform: translateY(100vh) rotate(0deg);
                opacity: 0;
            }
            10% { opacity: 1; }
            90% { opacity: 1; }
            100% {
                transform: translateY(-100vh) rotate(720deg);
                opacity: 0;
            }
        }

        /* Page Wrapper */
        .page-wrapper {
            min-height: 100vh;
            padding: 2rem 1rem;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        /* Glass Card */
        .glass-card {
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            border-radius: 2rem;
            box-shadow: 
                0 25px 50px -12px rgba(0, 0, 0, 0.25),
                0 0 0 1px var(--glass-border),
                inset 0 1px 0 rgba(255, 255, 255, 0.5);
            padding: 2.5rem;
            position: relative;
            overflow: hidden;
        }

        .glass-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.8), transparent);
        }

        /* Logo Animation */
        .logo-container {
            position: relative;
        }

        .logo-circle {
            width: 70px;
            height: 70px;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 32px;
            box-shadow: 
                0 10px 40px rgba(99, 102, 241, 0.4),
                0 0 0 4px rgba(99, 102, 241, 0.1);
            animation: pulse 3s ease-in-out infinite;
            position: relative;
            z-index: 1;
        }

        .logo-circle::after {
            content: '';
            position: absolute;
            inset: -4px;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            z-index: -1;
            opacity: 0.3;
            animation: pulseRing 3s ease-in-out infinite;
        }

        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }

        @keyframes pulseRing {
            0%, 100% { transform: scale(1); opacity: 0.3; }
            50% { transform: scale(1.2); opacity: 0; }
        }

        /* Header */
        .header-title {
            font-size: 1.75rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--dark), var(--primary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.25rem;
        }

        .header-subtitle {
            font-size: 0.9rem;
            color: #64748b;
            line-height: 1.8;
        }

        /* Class Counter Badge */
        .class-badge {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 600;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
            animation: shimmer 2s ease-in-out infinite;
        }

        @keyframes shimmer {
            0%, 100% { box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3); }
            50% { box-shadow: 0 4px 25px rgba(99, 102, 241, 0.5); }
        }

        /* Section Titles */
        .section-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--dark);
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .section-title i {
            color: var(--primary);
            font-size: 1.2rem;
        }

        /* Form Styles */
        .form-control {
            border: 2px solid #e2e8f0;
            border-radius: 1rem;
            padding: 1rem 1.25rem;
            font-size: 1rem;
            transition: all 0.3s ease;
            background: rgba(255, 255, 255, 0.8);
        }

        .form-control:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.1);
            background: white;
            transform: translateY(-2px);
        }

        .form-label {
            font-weight: 600;
            color: var(--dark);
            margin-bottom: 0.5rem;
            font-size: 0.95rem;
        }

        /* Buttons */
        .btn-primary-custom {
            background: linear-gradient(135deg, var(--primary), var(--primary-dark));
            border: none;
            padding: 1rem 2rem;
            font-size: 1.1rem;
            font-weight: 600;
            border-radius: 1rem;
            color: white;
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        }

        .btn-primary-custom::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s ease;
        }

        .btn-primary-custom:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4);
        }

        .btn-primary-custom:hover::before {
            left: 100%;
        }

        .btn-primary-custom:active {
            transform: translateY(-1px);
        }

        .btn-secondary-custom {
            background: white;
            border: 2px solid var(--primary);
            padding: 1rem 2rem;
            font-size: 1.1rem;
            font-weight: 600;
            border-radius: 1rem;
            color: var(--primary);
            transition: all 0.3s ease;
        }

        .btn-secondary-custom:hover {
            background: var(--primary);
            color: white;
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(99, 102, 241, 0.3);
        }

        /* Pills */
        .pill {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            background: linear-gradient(135deg, #f1f5f9, #e2e8f0);
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 500;
            margin: 0.25rem;
            transition: all 0.3s ease;
            border: 1px solid transparent;
        }

        .pill-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            animation: dotPulse 2s ease-in-out infinite;
        }

        @keyframes dotPulse {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.3); opacity: 0.7; }
        }

        .example-chip {
            cursor: pointer;
            background: white;
            border: 1px solid #e2e8f0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }

        .example-chip:hover {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 8px 20px rgba(99, 102, 241, 0.3);
            border-color: transparent;
        }

        .example-chip:hover .pill-dot {
            background: white;
        }

        /* Divider */
        .divider {
            height: 2px;
            background: linear-gradient(90deg, transparent, #e2e8f0, transparent);
            margin: 2rem 0;
            position: relative;
        }

        .divider::after {
            content: '✦';
            position: absolute;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
            background: white;
            padding: 0 1rem;
            color: var(--primary);
            font-size: 1.2rem;
        }

        /* Result Badge */
        .result-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.75rem;
            background: linear-gradient(135deg, var(--success), #059669);
            color: white;
            padding: 1rem 2rem;
            border-radius: 1rem;
            font-size: 1.2rem;
            font-weight: 600;
            box-shadow: 0 10px 40px rgba(16, 185, 129, 0.3);
            animation: resultPop 0.5s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        }

        @keyframes resultPop {
            0% { transform: scale(0); opacity: 0; }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); opacity: 1; }
        }

        .result-badge i {
            font-size: 1.4rem;
        }

        /* Result Container */
        .result-container {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(6, 182, 212, 0.1));
            border: 1px solid rgba(16, 185, 129, 0.2);
            border-radius: 1.5rem;
            padding: 1.5rem;
            text-align: center;
            animation: fadeInUp 0.5s ease;
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        /* OCR Box */
        .ocr-box {
            background: linear-gradient(135deg, #f8fafc, #f1f5f9);
            border: 1px solid #e2e8f0;
            border-radius: 1rem;
            padding: 1.5rem;
            font-size: 1rem;
            white-space: pre-wrap;
            line-height: 2;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
        }

        /* Uploaded Image */
        .uploaded-image {
            max-width: 100%;
            width: 500px;
            border-radius: 1.5rem;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
            border: 4px solid white;
            animation: imageReveal 0.6s ease;
        }

        @keyframes imageReveal {
            from {
                opacity: 0;
                transform: scale(0.9) rotate(-2deg);
            }
            to {
                opacity: 1;
                transform: scale(1) rotate(0deg);
            }
        }

        /* Footer */
        .footer {
            margin-top: 2rem;
            padding-top: 1.5rem;
            border-top: 1px solid #e2e8f0;
            text-align: center;
            color: #94a3b8;
            font-size: 0.85rem;
        }

        .footer-tech {
            display: flex;
            justify-content: center;
            gap: 1rem;
            flex-wrap: wrap;
            margin-top: 0.75rem;
        }

        .tech-badge {
            background: #f1f5f9;
            padding: 0.35rem 0.75rem;
            border-radius: 0.5rem;
            font-size: 0.75rem;
            font-weight: 500;
            color: #64748b;
            display: flex;
            align-items: center;
            gap: 0.35rem;
        }

        /* File Input Styling */
        .file-upload-wrapper {
            position: relative;
            background: linear-gradient(135deg, #f8fafc, #f1f5f9);
            border: 2px dashed #cbd5e1;
            border-radius: 1rem;
            padding: 2rem;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
        }

        .file-upload-wrapper:hover {
            border-color: var(--primary);
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.05), rgba(6, 182, 212, 0.05));
        }

        .file-upload-wrapper input[type="file"] {
            position: absolute;
            inset: 0;
            opacity: 0;
            cursor: pointer;
        }

        .file-upload-icon {
            font-size: 2.5rem;
            color: var(--primary);
            margin-bottom: 0.75rem;
        }

        .file-upload-text {
            color: #64748b;
            font-size: 0.9rem;
        }

        /* Loading Spinner */
        .loading-spinner {
            display: none;
            width: 24px;
            height: 24px;
            border: 3px solid rgba(255,255,255,0.3);
            border-top-color: white;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .btn-loading .loading-spinner {
            display: inline-block;
        }

        .btn-loading .btn-text {
            opacity: 0.7;
        }

        /* Stagger Animation for Pills */
        .stagger-item {
            opacity: 0;
            animation: staggerIn 0.4s ease forwards;
        }

        @keyframes staggerIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        /* Responsive */
        @media (max-width: 768px) {
            .glass-card {
                padding: 1.5rem;
                border-radius: 1.5rem;
            }
            
            .header-title {
                font-size: 1.4rem;
            }
            
            .logo-circle {
                width: 56px;
                height: 56px;
                font-size: 26px;
            }
        }

        /* Reduced Motion */
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            }
        }

        /* Confetti Animation */
        .confetti-container {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 9999;
            overflow: hidden;
        }

        .confetti {
            position: absolute;
            width: 10px;
            height: 10px;
            opacity: 0;
        }

        /* Typing indicator */
        .typing-indicator {
            display: inline-flex;
            gap: 4px;
            padding: 0.5rem;
        }

        .typing-indicator span {
            width: 8px;
            height: 8px;
            background: var(--primary);
            border-radius: 50%;
            animation: typing 1.4s infinite ease-in-out both;
        }

        .typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
        .typing-indicator span:nth-child(2) { animation-delay: -0.16s; }

        @keyframes typing {
            0%, 80%, 100% { transform: scale(0.6); opacity: 0.6; }
            40% { transform: scale(1); opacity: 1; }
        }

        /* Glow effect on focus */
        .glow-on-focus:focus-within {
            box-shadow: 0 0 30px rgba(99, 102, 241, 0.2);
        }
    </style>
</head>
<body>
    <!-- Animated Background -->
    <div class="bg-animated"></div>
    
    <!-- Floating Orbs -->
    <div class="orb orb-1"></div>
    <div class="orb orb-2"></div>
    <div class="orb orb-3"></div>
    
    <!-- Particles -->
    <div class="particles" id="particles"></div>
    
    <!-- Confetti Container -->
    <div class="confetti-container" id="confetti-container"></div>

    <div class="page-wrapper">
        <div class="container">
            <div class="row justify-content-center">
                <div class="col-12 col-lg-9 col-xl-8">
                    <div class="glass-card animate__animated animate__fadeInUp">
                        <!-- Header -->
                        <div class="d-flex align-items-center justify-content-between mb-4 flex-wrap gap-3">
                            <div class="d-flex align-items-center gap-3">
                                <div class="logo-container">
                                    <div class="logo-circle">
                                        <span>📰</span>
                                    </div>
                                </div>
                                <div>
                                    <h1 class="header-title">Urdu News Headline Classifier</h1>
                                    <p class="header-subtitle">
                                        اردو ہیڈ لائن لکھیں یا خبر کی تصویر اپلوڈ کریں، ماڈل خود بخود category پیش کرے گا
                                    </p>
                                </div>
                            </div>
                            <div class="class-badge">
                                <i class="fas fa-layer-group me-1"></i>
                                {{ classes|length }} Classes
                            </div>
                        </div>

                        <!-- Text Classification Section -->
                        <div class="section-title">
                            <i class="fas fa-keyboard"></i>
                            <span>Text Classification</span>
                        </div>
                        
                        <form method="post" id="text-form" class="glow-on-focus">
                            <input type="hidden" name="form_type" value="text">
                            <div class="mb-3">
                                <label for="headline" class="form-label">
                                    <i class="fas fa-pen-fancy me-2 text-primary"></i>
                                    اردو ہیڈ لائن لکھیں
                                </label>
                                <textarea
                                    class="form-control"
                                    name="headline"
                                    id="headline"
                                    required
                                    rows="4"
                                    placeholder="مثال: پاکستان نے بھارت کو سنسنی خیز میچ میں شکست دے دی"
                                >{{ headline or "" }}</textarea>
                            </div>

                            <div class="d-grid mb-3">
                                <button type="submit" class="btn btn-primary-custom" id="text-submit-btn">
                                    <span class="btn-text">
                                        <i class="fas fa-magic me-2"></i>
                                        Classify Headline
                                    </span>
                                    <div class="loading-spinner"></div>
                                </button>
                            </div>
                        </form>

                        {% if prediction %}
                        <div class="result-container mb-3 animate__animated animate__bounceIn">
                            <p class="mb-2 text-muted">Predicted Category:</p>
                            <div class="result-badge">
                                <i class="fas fa-check-circle"></i>
                                {{ prediction }}
                            </div>
                        </div>
                        {% endif %}

                        <!-- Available Classes -->
                        <div class="mb-3">
                            <div class="section-title" style="font-size: 0.95rem;">
                                <i class="fas fa-tags"></i>
                                <span>Available Categories</span>
                            </div>
                            <div class="d-flex flex-wrap">
                                {% for cls in classes %}
                                <span class="pill stagger-item" style="animation-delay: {{ loop.index * 0.05 }}s">
                                    <span class="pill-dot"></span>
                                    {{ cls }}
                                </span>
                                {% endfor %}
                            </div>
                        </div>

                        <!-- Example Headlines -->
                        <div class="mb-4">
                            <div class="section-title" style="font-size: 0.95rem;">
                                <i class="fas fa-lightbulb"></i>
                                <span>Try Example Headlines</span>
                            </div>
                            <div class="d-flex flex-wrap">
                                <span class="pill example-chip stagger-item" data-class="business" style="animation-delay: 0.1s">
                                    <i class="fas fa-briefcase"></i>
                                    business
                                </span>
                                <span class="pill example-chip stagger-item" data-class="entertainment" style="animation-delay: 0.15s">
                                    <i class="fas fa-film"></i>
                                    entertainment
                                </span>
                                <span class="pill example-chip stagger-item" data-class="science-technology" style="animation-delay: 0.2s">
                                    <i class="fas fa-flask"></i>
                                    science-technology
                                </span>
                                <span class="pill example-chip stagger-item" data-class="sports" style="animation-delay: 0.25s">
                                    <i class="fas fa-futbol"></i>
                                    sports
                                </span>
                                <span class="pill example-chip stagger-item" data-class="world" style="animation-delay: 0.3s">
                                    <i class="fas fa-globe"></i>
                                    world
                                </span>
                            </div>
                        </div>

                        <div class="divider"></div>

                        <!-- Image Classification Section -->
                        <div class="section-title">
                            <i class="fas fa-image"></i>
                            <span>Image Classification (OCR)</span>
                        </div>

                        <form method="post" enctype="multipart/form-data" id="image-form">
                            <input type="hidden" name="form_type" value="image">
                            <div class="mb-3">
                                <label class="form-label">
                                    <i class="fas fa-cloud-upload-alt me-2 text-primary"></i>
                                    خبر کی تصویر اپلوڈ کریں
                                </label>
                                <div class="file-upload-wrapper">
                                    <input type="file" id="image_file" name="image_file" accept="image/*">
                                    <div class="file-upload-icon">
                                        <i class="fas fa-cloud-upload-alt"></i>
                                    </div>
                                    <p class="file-upload-text mb-0">
                                        <strong>Click to upload</strong> or drag and drop<br>
                                        <small>PNG, JPG, JPEG (Max 10MB)</small>
                                    </p>
                                </div>
                            </div>
                            <div class="d-grid mb-3">
                                <button type="submit" class="btn btn-secondary-custom" id="image-submit-btn">
                                    <span class="btn-text">
                                        <i class="fas fa-search me-2"></i>
                                        Extract & Classify
                                    </span>
                                    <div class="loading-spinner"></div>
                                </button>
                            </div>
                        </form>

                        {% if image_prediction %}
                        <div class="result-container mb-3 animate__animated animate__bounceIn">
                            <p class="mb-2 text-muted">Image Predicted Category:</p>
                            <div class="result-badge">
                                <i class="fas fa-check-circle"></i>
                                {{ image_prediction }}
                            </div>
                        </div>
                        {% endif %}

                        {% if extracted_text %}
                        <div class="mb-3 animate__animated animate__fadeIn">
                            <div class="section-title" style="font-size: 0.95rem;">
                                <i class="fas fa-file-alt"></i>
                                <span>Extracted OCR Text</span>
                            </div>
                            <div class="ocr-box">{{ extracted_text }}</div>
                        </div>
                        {% endif %}

                        {% if image_url %}
                        <div class="text-center mb-3">
                            <img src="{{ image_url }}" alt="Uploaded image" class="uploaded-image">
                        </div>
                        {% endif %}

                        <!-- Footer -->
                        <div class="footer">
                            <p class="mb-1">
                                <i class="fas fa-graduation-cap me-1"></i>
                                ML Project - Urdu News Headline Classification
                            </p>
                            <div class="footer-tech">
                                <span class="tech-badge">
                                    <i class="fab fa-python"></i>
                                    Python
                                </span>
                                <span class="tech-badge">
                                    <i class="fas fa-brain"></i>
                                    Naive Bayes
                                </span>
                                <span class="tech-badge">
                                    <i class="fas fa-eye"></i>
                                    Google Vision OCR
                                </span>
                                <span class="tech-badge">
                                    <i class="fab fa-python"></i>
                                    Flask
                                </span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
    
    <!-- Anime.js for advanced animations -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/animejs/3.2.2/anime.min.js"></script>

    <script>
        // Example Headlines Data
        const EXAMPLES = {
            "business": [
                "حکومت کی اقتصادی ٹیم نے نئی مالیاتی پالیسی منظور کر لی، کاروباری حلقوں نے محتاط ردعمل دیا۔",
                "اسٹاک مارکیٹ میں تیزی، انڈیکس پانچ سو پوائنٹس بڑھ گیا اور سرمایہ کاروں کا اعتماد بحال ہونے لگا۔",
                "ڈالر کی قدر میں کمی کے بعد درآمدی اشیا کی قیمتوں میں کمی کا امکان ظاہر کیا جا رہا ہے۔",
                "چھوٹے تاجروں کے لیے آسان قرض اسکیم متعارف، بینکوں نے درخواستوں کی وصولی شروع کر دی۔",
                "عالمی منڈی میں تیل کی قیمت میں اضافہ، مقامی سطح پر پیٹرولیم مصنوعات مہنگی ہونے کا خدشہ۔"
            ],
            "entertainment": [
                "معروف اداکارہ کی نئی فلم ریلیز ہوتے ہی سینما گھروں میں رش لگ گیا، پروڈیوسر نے خوشی کا اظہار کیا۔",
                "ڈرامہ سیریل کی آخری قسط نے ریکارڈ ریٹنگ حاصل کر لی، سوشل میڈیا پر مداحوں کے دلچسپ تبصرے جاری ہیں۔",
                "مشہور گلوکار نے ورلڈ ٹور کا اعلان کر دیا، پاکستان میں بھی کنسرٹ منعقد کرنے کی تیاری مکمل۔",
                "ایوارڈ شو میں سال کی بہترین اداکارہ اور بہترین ڈرامہ سیریل کا اعلان، پرستاروں میں جوش و خروش۔",
                "آن لائن اسٹریمنگ پلیٹ فارم پر نئی ویب سیریز نے چند ہی دنوں میں بڑی تعداد میں ناظرین حاصل کر لیے۔"
            ],
            "science-technology": [
                "مقامی یونیورسٹی کے طلبہ نے مصنوعی ذہانت پر مبنی اردو چیٹ بوٹ تیار کیا جو طلبہ کے سوالات کے جواب دیتا ہے۔",
                "دنیا کی بڑی ٹیک کمپنی نے نیا اسمارٹ فون متعارف کرا دیا جس میں بہتر کیمرہ اور بیٹری ٹائم شامل ہے۔",
                "سائنس دانوں نے شمسی توانائی سے چلنے والا نیا سسٹم تیار کیا جس سے بجلی کے اخراجات میں واضح کمی ہو گی۔",
                "حکومت نے ملک بھر میں فائیو جی ٹیکنالوجی کے ٹرائلز شروع کر دیے، صارفین تیز رفتار انٹرنیٹ کے منتظر۔",
                "ریسرچ رپورٹ کے مطابق روبوٹکس اور آٹومیشن آئندہ دس سال میں ہزاروں نئی ملازمتیں پیدا کریں گے۔"
            ],
            "sports": [
                "ٹی ٹوئنٹی سیریز کے فیصلہ کن میچ میں کپتان کی نصف سنچری اور شاندار بولنگ سے ٹیم نے کامیابی سمیٹ لی۔",
                "ایشیا کپ کے لیے قومی اسکواڈ کا اعلان، نوجوان کھلاڑیوں کی شمولیت پر شائقین کرکٹ نے خوشی کا اظہار کیا۔",
                "لیگ میچ میں فٹبال کلب نے تین صفر سے فتح حاصل کر کے پوائنٹس ٹیبل پر پہلی پوزیشن حاصل کر لی۔",
                "قومی ہاکی ٹیم نے روایتی حریف کے خلاف شاندار کارکردگی دکھاتے ہوئے سیریز برابر کر دی۔",
                "اولمپکس کی تیاری کے لیے کھلاڑیوں کا تربیتی کیمپ شروع، کوچز نے فٹنس کو اولین ترجیح قرار دیا۔"
            ],
            "world": [
                "اقوام متحدہ کے اجلاس میں عالمی معاشی بحران اور مہنگائی کے خلاف مشترکہ حکمت عملی پر غور کیا گیا۔",
                "یورپی ممالک نے ماحولیاتی تبدیلی سے نمٹنے کے لیے نئے معاہدے پر دستخط کر دیے، ماہرین نے اقدام کو سراہا۔",
                "مشرق وسطیٰ میں کشیدگی بڑھنے پر عالمی طاقتوں نے فریقین سے تحمل اور مذاکرات پر زور دیا ہے۔",
                "عالمی ادارہ صحت نے نئی وبا کے خدشات پر خبردار کرتے ہوئے ممالک کو احتیاطی اقدامات بڑھانے کا مشورہ دیا۔",
                "سرحدی تنازع پر دو ملکوں کے درمیان بات چیت کا نیا دور شروع، سفارتی ذرائع نے پیش رفت کو حوصلہ افزا قرار دیا۔"
            ]
        };

        const exampleIndices = {};

        // Initialize example indices
        Object.keys(EXAMPLES).forEach(key => {
            exampleIndices[key] = 0;
        });

        // Example chip click handlers
        document.querySelectorAll(".example-chip").forEach(chip => {
            chip.addEventListener("click", function() {
                const cls = this.getAttribute("data-class");
                const list = EXAMPLES[cls];
                if (!list || list.length === 0) return;

                const i = exampleIndices[cls] % list.length;
                const txt = list[i];

                const textarea = document.getElementById("headline");
                
                // Animate the textarea
                anime({
                    targets: textarea,
                    scale: [1, 1.02, 1],
                    duration: 300,
                    easing: 'easeInOutQuad'
                });
                
                textarea.value = txt;
                textarea.focus();

                exampleIndices[cls] = (i + 1) % list.length;
                
                // Animate the clicked chip
                anime({
                    targets: this,
                    scale: [1, 0.95, 1],
                    duration: 200,
                    easing: 'easeInOutQuad'
                });
            });
        });

        // Create particles
        function createParticles() {
            const container = document.getElementById('particles');
            const particleCount = 30;
            
            for (let i = 0; i < particleCount; i++) {
                const particle = document.createElement('div');
                particle.className = 'particle';
                particle.style.left = Math.random() * 100 + '%';
                particle.style.animationDelay = Math.random() * 15 + 's';
                particle.style.animationDuration = (15 + Math.random() * 10) + 's';
                container.appendChild(particle);
            }
        }

        // Confetti effect
        function createConfetti() {
            const container = document.getElementById('confetti-container');
            const colors = ['#6366f1', '#06b6d4', '#10b981', '#f59e0b', '#ec4899'];
            
            for (let i = 0; i < 50; i++) {
                const confetti = document.createElement('div');
                confetti.className = 'confetti';
                confetti.style.left = Math.random() * 100 + '%';
                confetti.style.background = colors[Math.floor(Math.random() * colors.length)];
                confetti.style.borderRadius = Math.random() > 0.5 ? '50%' : '0';
                container.appendChild(confetti);
                
                anime({
                    targets: confetti,
                    translateY: [-20, window.innerHeight + 20],
                    translateX: () => anime.random(-100, 100),
                    rotate: () => anime.random(0, 360),
                    opacity: [1, 0],
                    duration: () => anime.random(2000, 4000),
                    delay: () => anime.random(0, 500),
                    easing: 'easeOutQuad',
                    complete: () => confetti.remove()
                });
            }
        }

        // Initialize particles
        createParticles();

        // Trigger confetti on result
        {% if prediction or image_prediction %}
        setTimeout(createConfetti, 300);
        {% endif %}

        // File input styling
        const fileInput = document.getElementById('image_file');
        const fileWrapper = fileInput.closest('.file-upload-wrapper');
        
        fileInput.addEventListener('change', function() {
            if (this.files.length > 0) {
                fileWrapper.querySelector('.file-upload-text').innerHTML = 
                    '<strong>' + this.files[0].name + '</strong><br><small>Ready to upload</small>';
                
                anime({
                    targets: fileWrapper,
                    scale: [1, 1.02, 1],
                    borderColor: '#10b981',
                    duration: 300,
                    easing: 'easeInOutQuad'
                });
            }
        });

        // Form submit loading state
        document.querySelectorAll('form').forEach(form => {
            form.addEventListener('submit', function() {
                const btn = this.querySelector('button[type="submit"]');
                btn.classList.add('btn-loading');
                btn.disabled = true;
            });
        });

        // Hover effects for cards
        const glassCard = document.querySelector('.glass-card');
        glassCard.addEventListener('mousemove', (e) => {
            const rect = glassCard.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            glassCard.style.background = `
                radial-gradient(circle at ${x}px ${y}px, rgba(99, 102, 241, 0.03), transparent 50%),
                rgba(255, 255, 255, 0.95)
            `;
        });

        glassCard.addEventListener('mouseleave', () => {
            glassCard.style.background = 'rgba(255, 255, 255, 0.95)';
        });
    </script>
</body>
</html>
"""

# =========================================================
# 4. Flask route using NB for text + image
# =========================================================

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_prediction = None
    headline = None
    extracted_text = None
    image_url = None

    if request.method == "POST":
        form_type = request.form.get("form_type", "text")

        if form_type == "text":
            headline = request.form.get("headline", "").strip()
            if headline:
                prediction = predict_headline(headline)

        elif form_type == "image":
            file = request.files.get("image_file")
            if file and file.filename:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(filepath)
                image_url = url_for("static", filename=f"uploads/{filename}")
                
                try:
                    ocr_text, label = predict_from_image(filepath)
                    extracted_text = ocr_text or "No text detected by OCR."
                    image_prediction = label if label is not None else "Could not classify"
                except Exception as e:
                    extracted_text = f"OCR error: {e}"
                    image_prediction = "OCR failed"

    return render_template_string(
        HTML_TEMPLATE,
        classes=CLASSES,
        prediction=prediction,
        image_prediction=image_prediction,
        headline=headline,
        extracted_text=extracted_text,
        image_url=image_url
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
