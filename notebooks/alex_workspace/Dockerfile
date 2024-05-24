FROM python:3.10.6-buster

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip --no-cache-dir -r requirements.txt

# Command to run FastAPI and Streamlit
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT]
#& streamlit run streamlit_app.py"
