# FROM python:3.10.6-buster
FROM ubuntu:22.04

RUN apt-get update
RUN apt install -y python3.10
RUN apt install -y python3-pip

# Set the working directory
WORKDIR /app

# Copy Files
COPY requirements.txt requirements.txt

# Tesseract
# RUN add-apt-repository -y ppa:alex-p/tesseract-ocr
# RUN apt-get update
RUN apt-get install -y tesseract-ocr
RUN apt-get install -y libtesseract-dev
RUN apt-get install -y tesseract-ocr-eng
RUN apt-get install -y tesseract-ocr-jpn
RUN apt-get install -y tesseract-ocr-osd 

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY api api
COPY pokedex pokedex

ENV PYTHONPATH="${PYTHONPATH}:/app"

# # Expose the port Streamlit runs on
# EXPOSE 8501

# Command to run the Streamlit app
CMD uvicorn api.main:app --host 0.0.0.0 --port $PORT
