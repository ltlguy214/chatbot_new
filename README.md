# 🎵 VMusicAI – AI-powered V-Pop Analysis & Recommendation System

> Graduation Project – University of Information Technology (UIT – VNU-HCM)

An end-to-end AI/Data Science platform that analyzes Vietnamese V-Pop songs using Machine Learning, Natural Language Processing (PhoBERT), Explainable AI and Large Language Models (Gemini).

The system supports music analysis, hit song prediction, semantic search and AI-powered conversational assistance through an interactive Streamlit application.

<p align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange)
![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-blue)
![Supabase](https://img.shields.io/badge/Supabase-green)
![PhoBERT](https://img.shields.io/badge/NLP-PhoBERT-success)
![Gemini API](https://img.shields.io/badge/LLM-Gemini_API-purple)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)


# 📷 Demonstration

## Home Page
---
<p align="center">

<img src="images/home.png" width="900">

</p>

## Recommendation

<p align="center">

<img src="images/recommendation.png" width="900">

</p>

## Song Analysis

<p align="center">

<img src="images/song_analysis.png" width="900">

</p>

---
# 🎯 Objectives

The objectives of VMusicAI are to:

- Analyze Vietnamese V-Pop songs using Machine Learning techniques.
- Predict hit potential and popularity trends.
- Provide intelligent song analysis through Natural Language Processing.
- Support semantic retrieval using vector databases.
- Deliver an AI-powered conversational assistant for music exploration.

# 📖 Project Overview

VMusicAI is an end-to-end AI/Data Science platform developed as a graduation project at the University of Information Technology (UIT – VNU-HCM).

The project aims to analyze Vietnamese V-Pop songs through multiple Machine Learning tasks while providing intelligent recommendations and conversational support using Natural Language Processing and Large Language Models.

Unlike conventional music recommendation systems, VMusicAI combines predictive analytics, explainable AI, semantic search and an AI chatbot into a unified platform, allowing users to better understand song characteristics and prediction results.

The system covers the complete AI workflow, including:

- Data Collection
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Machine Learning
- NLP
- Explainable AI
- Deployment

# 📊 Project Highlights

- ✅ End-to-end AI/Data Science workflow
- 🎵 Processed 7,665 Vietnamese V-Pop songs
- 🤖 Integrated Machine Learning, NLP and LLM into a unified platform
- 🧠 PhoBERT for Vietnamese intent classification
- 💬 Gemini-powered AI chatbot with contextual conversations
- 🔍 Semantic search using PostgreSQL + pgvector
- 📊 Explainable AI with SHAP
- ⚙ Automated hyperparameter optimization using Optuna
- 🌐 Interactive Streamlit web application
- 🎓 Graduation Project at the University of Information Technology (UIT – VNU-HCM)

# 🚀 Key Features

- 🎯 Hit Song Prediction
- 📈 Popularity Prediction
- 🎼 Genre Classification
- 😊 Emotion Classification
- 📊 Music Style Clustering
- 🤖 AI Chatbot powered by Gemini
- 🔍 Semantic Search using pgvector
- 🧠 PhoBERT Intent Classification
- 📊 Explainable AI with SHAP
- ⚙ Hyperparameter Optimization using Optuna

# 📂 Dataset

The project was developed using a curated Vietnamese V-Pop dataset containing metadata, audio-related features and song information collected from public music platforms.

| Dataset | Size |
|----------|------|
| Songs | 7,665 |
| Genres | Multiple |
| Artists | 1,668 |
| Lyrics | Vietnamese |
| Numerical Features | 75 | 
| Binary Features | 24 |
| OHE Features | 3 |
| Evaluation Scenarios | 388 |
| Date | 2010/01/01 - 2026/03/18 |

The dataset was preprocessed through data cleaning, feature engineering and exploratory data analysis before training Machine Learning models.
---

# 🏗️ System Architecture

                           VMusicAI Architecture

┌───────────────────────────────────────────────────────────────────────┐
│                           Data Sources                                │
│  • Spotify API                                                        │
│  • Genius Lyrics                                                      │
│  • V-Pop Metadata                                                     │
└───────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│                     Data Collection & Preprocessing                   │
│  • Data Cleaning                                                      │
│  • Missing Value Handling                                             │
│  • Feature Engineering                                                │
│  • Exploratory Data Analysis                                          │
└───────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│                     Machine Learning Module                           │
│  Classification                                                       │
│  Regression                                                           │
│  Clustering                                                           │
│  Model Evaluation                                                     │
│  SHAP                                                                 │
│  Optuna                                                               │
└───────────────────────────────────────────────────────────────────────┘
                 │                           │
                 ▼                           ▼
┌─────────────────────────────┐   ┌─────────────────────────────┐
│       NLP Module            │   │      Recommendation         │
│ PhoBERT                     │   │ Semantic Search             │
│ Intent Classification       │   │ pgvector                    │
│                             │   │ Similarity Ranking          │
└─────────────────────────────┘   └─────────────────────────────┘
                 │                           │
                 └───────────────┬───────────┘
                                 ▼
┌───────────────────────────────────────────────────────────────────────┐
│                       Gemini AI Chatbot                               │
│ Context-aware Conversation                                            │
│ Prompt Engineering                                                    │
│ Recommendation Explanation                                            │
└───────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│                   Streamlit Web Application                           │
└───────────────────────────────────────────────────────────────────────┘

The architecture consists of four major modules:

- Data Processing
- Machine Learning
- NLP & LLM
- Web Application
---


# 🔄 Machine Learning Pipeline

The project follows a complete end-to-end Machine Learning workflow.

```text
Data Collection
      │
      ▼
Data Cleaning
      │
      ▼
Exploratory Data Analysis (EDA)
      │
      ▼
Feature Engineering
      │
      ▼
Model Training
      │
      ▼
Hyperparameter Optimization
      │
      ▼
Model Evaluation
      │
      ▼
Explainability (SHAP)
      │
      ▼
Deployment (Streamlit)
```

---

# 🤖 Machine Learning Tasks

VMusicAI consists of five Machine Learning tasks for supporting V-Pop music analysis.

| Task | Description |
|------|-------------|
| 🎯 Hit Song Classification | Predict whether a song has hit potential |
| 📈 Popularity Prediction | Estimate song popularity using regression models |
| 🎼 Music Style Clustering | Group songs based on audio and metadata similarity |
| 😊 Emotion Classification | Identify emotional characteristics of songs |
| 🎤 Genre Classification | Predict music genres from extracted features |

---

# 🧠 NLP & Large Language Models

The NLP module enables intelligent conversations and semantic understanding.

### PhoBERT

- Vietnamese Intent Classification
- User query understanding
- Context-aware intent detection

### Gemini API

- Conversational AI
- Song explanation
- Recommendation reasoning
- Natural language interaction

### Vector Search

- PostgreSQL + pgvector
- Embedding similarity search
- Semantic retrieval
- Context enhancement for chatbot

---

# 📊 Explainable AI

To improve model transparency and interpretability, SHAP (SHapley Additive exPlanations) is integrated into the prediction workflow.

Features include:

- Global Feature Importance
- Local Prediction Explanation
- SHAP Summary Plot
- Individual Prediction Interpretation

This enables users to understand why a prediction is generated rather than receiving only the final output.

---

# ⚙️ Hyperparameter Optimization

The project utilizes **Optuna** to automatically search for optimal hyperparameters.

Optimization objectives include:

- Improving model accuracy
- Reducing overfitting
- Selecting optimal parameter combinations
- Increasing model robustness

---

# 📈 Experimental Results

| Module | Metric | Result |
|----------|----------|---------|
| Intent Classification | Accuracy | 96.90% |
| Intent Classification | Macro F1-score | 97.32% |
| Chatbot Evaluation | HitRate@1 | 92.25% |
| Chatbot Evaluation | Mean Reciprocal Rank | 92.25% |
| End-to-End Testing | Test Scenarios | 388 |

---

# 💻 Technology Stack

### Programming

- Python

### Data Science

- Pandas
- NumPy
- Scikit-learn

### Natural Language Processing

- PhoBERT
- Gemini API

### Explainable AI

- SHAP
- Optuna

### Database

- PostgreSQL
- Supabase
- pgvector

### Visualization

- Matplotlib
- Streamlit

### Version Control

- Git
- GitHub

---

# 📁 Project Structure

```text
VMusicAI
│
├── chatbot/
├── data/
├── models/
├── scripts/
├── dev/
├── images/
├── docs/
├── README.md
├── requirements.txt
└── LICENSE
```
# 🚀 Getting Started

## Prerequisites

Before running the project, ensure your environment meets the following requirements:

- Python 3.12+
- PostgreSQL
- Supabase Account
- Gemini API Key
- Git
- Streamlit

---

## Installation

Clone the repository

```bash
git clone https://github.com/ltlguy214/VMusicAI.git

cd VMusicAI
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the virtual environment

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file inside the chatbot directory.

```env
SUPABASE_URL=YOUR_SUPABASE_URL

SUPABASE_KEY=YOUR_SUPABASE_KEY

GOOGLE_API_KEY=YOUR_GEMINI_API_KEY

MODELS_PREFER_STORAGE=true
```

---

## Run the Application

Launch Streamlit

```bash
streamlit run chatbot/app_chatbot.py
```

The application will be available at

```
http://localhost:8501
```

---

## Machine Learning

✔ Hit Song Classification
✔ Popularity Prediction
✔ Genre Classification
✔ Emotion Classification
✔ Music Style Clustering

---

## NLP

PhoBERT Intent Classification

Accuracy: 96.90%
Macro F1-score: 97.32%

---

## Chatbot Evaluation

388 testing scenarios

HitRate@1: 92.25%
MRR: 92.25%
---

## Explainability

The prediction process can be interpreted using SHAP.

Supported visualizations include

- Feature Importance
- SHAP Summary Plot
- Waterfall Plot
- Force Plot

This improves model transparency and supports trustworthy AI.

# 🚀 Future Improvements

- Enhance recommendation quality using hybrid recommendation techniques.
- Expand the Vietnamese music dataset.
- Improve chatbot reasoning with Retrieval-Augmented Generation (RAG).
- Optimize deployment for cloud environments.
- Add user personalization and feedback learning.

---
# 👩‍💻 Author

**Lê Trần Thu Huyền**

Information Technology Student

University of Information Technology (UIT – VNU-HCM)

📧 huyen21042003@gmail.com

🔗 GitHub: https://github.com/Ltth2104

---

# 📄 License

This project is released under the MIT License.

---

# ⭐ Acknowledgements

This project was developed as a Graduation Project at the University of Information Technology (UIT – VNU-HCM).

Special thanks to my supervisors, teammates and the open-source community whose tools and libraries contributed to this project.
