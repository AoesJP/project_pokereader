FROM ubuntu:22.04

# Installing Python
RUN apt-get update
RUN apt install -y python3.10
RUN apt install -y python3-pip

# Set the working directory
WORKDIR /app

# Copy Files
COPY requirements.txt requirements.txt

# Tesseract
RUN apt-get install -y tesseract-ocr
RUN apt-get install -y libtesseract-dev
RUN apt-get install -y tesseract-ocr-eng
RUN apt-get install -y tesseract-ocr-jpn
RUN apt-get install -y tesseract-ocr-osd 

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copying pokedex and api
COPY api api
COPY pokedex pokedex

ENV PYTHONPATH="${PYTHONPATH}:/app"

# API Server
CMD uvicorn api.main:app --host 0.0.0.0 --port $PORT
