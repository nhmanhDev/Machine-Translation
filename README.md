# NEURAL MACHINE TRANSLATION (ENGLISH → VIETNAMESE)

## PROJECT OVERVIEW
This project implements a neural machine translation system from English to Vietnamese using a custom-built Transformer model with PyTorch. The backend is powered by FastAPI and the web interface is rendered using Jinja2. The entire system is Dockerized to facilitate easy deployment and scalability.

## FEATURES
- Preprocess text data (cleaning, tokenization, vocabulary mapping).
- Transformer-based encoder-decoder translation model.
- Custom greedy decoding method for generating translated sentences.
- FastAPI backend to serve translation requests.
- Web-based user interface for inputting text and viewing translations.
- Docker containerization for simplified deployment.

## REQUIREMENTS
- Python 3.10
- A modern web browser (Chrome, Firefox, etc.)

Install the dependencies using:
pip install -r requirements.txt
INSTALLATION
  1. Clone the Repository
  Clone the Git repository to your local machine by running: git clone https://github.com/nhmanhDev/Machine-Translation.git
  
    Then navigate to the project directory: cd Machine-Translation
  
  2. Setting up the Python Environment (Without Docker)
    Create a virtual environment: python -m venv venv
    Activate the virtual environment:
    
    On Linux/MacOS: source venv/bin/activate
    
    On Windows: venv\Scripts\activate
    Install dependencies: pip install -r requirements.txt
    
    Run the Application: uvicorn app:app --host 0.0.0.0 --port 8000
    Access the User Interface: Open your web browser and navigate to: http://localhost:8000

## USAGE
Translation Flow
Input: User enters an English sentence in the web interface.

Processing: The input text is preprocessed (cleaning, tokenization, and mapping to indices) and fed into the Transformer model.

Output: The model generates a Vietnamese translation token by token using greedy decoding until the <end> token is encountered.

Display: The translated sentence is then rendered in the web interface.

## TECHNOLOGIES USED
Python: Core programming language.

PyTorch: For building and training the Transformer model.

FastAPI: Web framework for building the API.

Uvicorn: ASGI server to run the FastAPI application.

Jinja2: Template engine for the web interface.

Docker: Containerization of the entire system.

Additional Libraries: NumPy, python-multipart.

## CONTRIBUTING
Contributions are welcome! Please fork the repository and submit a pull request with your enhancements.

## LICENSE
This project is licensed under the MIT License. See the LICENSE file for details.

## CONTACT
For questions or feedback, please contact: nhmanh.dev@gmail.com

## DOCKER IMAGE
You can also pull the pre-built Docker image directly using:

docker pull nhmanhdev/machine-translation:v1.0
