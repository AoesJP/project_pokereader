FROM python:3.10.6-buster

# Set the working directory
WORKDIR /app

# Copy Files
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY api/ /app/api
COPY pokedex/ /app/pokedex

# # Expose the port Streamlit runs on
# EXPOSE 8501

# Command to run the Uvicorn
CMD uvicorn api.main:app --host 0.0.0.0 --port $PORT
