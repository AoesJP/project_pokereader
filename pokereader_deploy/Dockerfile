FROM python:3.10.6-buster

# Set the working directory
WORKDIR /app

# Copy Files
COPY requirements.txt requirements.txt

# Tesseract
RUN apt-get update
RUN apt-get install -y tesseract-ocr

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
