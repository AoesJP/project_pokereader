from setuptools import setup, find_packages

setup(
    name='project_pokereader',
    version='1.0.0',
    description='A comprehensive Python package for machine learning model deployment with Streamlit, Docker, FastAPI, and Google Cloud Platform (GCP).',
    author='Yuri, Estelle, Emilia, Alex',
    author_email='alex.tm.chiu@gmail.com',
    packages=find_packages(),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=open('requirements.txt').readlines(),
)
