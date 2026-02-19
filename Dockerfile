FROM python:3.11-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY setup.py .
COPY data_service/ data_service/
COPY static/ static/
COPY config.json .
COPY run_dashboard.py .
COPY run_web_interface.py .
COPY main.py .

# Install CPU-only torch first (saves ~1.5GB vs full torch)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install package with all extras + yfinance (missing from setup.py but used)
RUN pip install --no-cache-dir -e ".[ai,visualization,web,realtime]" yfinance

# Download NLTK data
RUN python -c "\
import nltk; \
nltk.download('punkt'); \
nltk.download('punkt_tab'); \
nltk.download('stopwords'); \
nltk.download('wordnet'); \
nltk.download('averaged_perceptron_tagger'); \
nltk.download('averaged_perceptron_tagger_eng')"

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Download TextBlob corpora
RUN python -m textblob.download_corpora lite

# Create data and logs directories
RUN mkdir -p data logs

# Streamlit config for Docker (listen on 0.0.0.0)
RUN mkdir -p /root/.streamlit
COPY .streamlit/config.toml /root/.streamlit/config.toml

EXPOSE 8000 8501
