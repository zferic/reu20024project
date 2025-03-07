# Research LLM Backend

This is the backend service for the Research LLM application. It handles document retrieval and text generation.

## Setup

1. Clone the repository
2. Navigate to the backend directory:
   ```bash
   cd backend
   ```

3. Make the deployment script executable:
   ```bash
   chmod +x deploy.sh
   ```

4. Run the deployment script:
   ```bash
   ./deploy.sh
   ```

## Directory Structure

```
backend/
├── src/                    # Source code
│   ├── generator/         # Text generation
│   ├── models/           # Model implementations
│   ├── retriever/        # Document retrieval
│   ├── serialization/    # Data serialization
│   └── utils/            # Utility functions
├── papers/               # Research papers (not in git)
├── vectorstore/         # Vector store (not in git)
├── main.py              # Main FastAPI server
├── requirements.txt     # Python dependencies
└── deploy.sh           # Deployment script
```

## API Endpoints

- `POST /query`: Submit a question and get an answer
  ```json
  {
    "question": "Your question here"
  }
  ```

## Development

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the server:
   ```bash
   python main.py
   ```

## Notes

- The vector store and papers are not included in git. You'll need to add your papers to the `papers` directory.
- SSL certificates are generated automatically if they don't exist.
- The server runs on port 8000 with HTTPS enabled. 