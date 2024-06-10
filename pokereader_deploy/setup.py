from setuptools import setup, find_packages

setup(
    name='project_pokereader',
    version='1.0.1',
    description='A comprehensive Python package for machine learning model deployment with Streamlit, Docker, FastAPI, and Google Cloud Platform (GCP).',
    author='Yuri, Estelle, Emilia, Alex',
    author_email='alex.tm.chiu@gmail.com',
    packages=find_packages(),
    install_requires=[
        "fastapi==0.111.0",
        "fastapi-cli==0.0.4",
        "httpx==0.27.0",
        "joblib==1.4.2",
        "matplotlib==3.9.0",
        "matplotlib-inline==0.1.7",
        "numpy==1.26.4",
        "opencv-python==4.9.0.80",
        "opencv-python-headless==4.9.0.80",
        "pandas==2.2.2",
        "python-dateutil==2.9.0.post0",
        "pydantic==2.7.3",
        "pyocr==0.8.5",
        "pytesseract==0.3.10",
        "requests==2.31.0",
        "scikit-learn==1.4.2",
        "scipy==1.13.0",
        "setuptools==63.2.0",
        "streamlit==1.35.0",
        "tensorflow==2.16.1",
        "uvicorn==0.29.0"
        ])
