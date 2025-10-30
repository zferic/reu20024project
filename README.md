# Research LLM Project

A research paper question-answering system using LLMs and semantic search.

## Project Structure
```
.
├── frontend/               # React frontend application
│   ├── src/               # React source code
│   ├── public/            # Static assets
│   └── package.json       # Frontend dependencies
│
├── backend/               # Python backend services
│   ├── src/
│   │   ├── retriever/    # Document retrieval and embedding
│   │   ├── generator/    # LLM and text generation
│   │   ├── preprocessing/# Data preprocessing scripts
│   │   ├── models/       # Model definitions
│   │   ├── evaluation/   # Evaluation scripts
│   │   └── qastuff/      # QA dataset generation
│   ├── main.py           # Main FastAPI server
│   └── requirements.txt   # Python dependencies
│
└── papers/               # Research papers and documents
```

## Features
- Semantic search over research papers using FAISS
- LLM-powered question answering
- Conversation memory for context-aware responses
- React-based user interface
- FastAPI backend with async support

## Setup

### Backend Setup
1. Create and activate virtual environment:
   ```bash
   cd backend
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Unix:
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the backend server:
   ```bash
   python main.py
   ```

### Frontend Setup
1. Install dependencies:
   ```bash
   cd frontend
   npm install
   ```

2. Start the development server:
   ```bash
   npm start
   ```

## Usage
1. Access the web interface at `http://localhost:3000`
2. Upload research papers or use the existing corpus
3. Ask questions about the research papers
4. The system will retrieve relevant context and generate answers

## Development
- Backend API runs on `http://localhost:8000`
- Frontend development server runs on `http://localhost:3000`
- VLLM server runs on `http://localhost:8001`

## VLLM development
python -m vllm.entrypoints.api_server --model <model-name> --port 8001

# APACHE CONFIG/ CMDS
/etc/apache2/sites-available/prollm.conf # apache config file
sudo a2enmod proxy proxy_http rewrite headers
sudo systemctl reload apache2
sudo systemctl restart apache2

 sudo systemctl restart  reu20024-backend.service  #backend refresh
 sudo systemctl restart  reu20024-generator.service #generator refresh
