# NanoMatter Custom RAG App

## Overview
The NanoMatter Custom Retrieval-Augmented Generation (RAG) App is designed to enable intelligent query answering by leveraging document-based knowledge. It currently processes **PDF files** to generate contextually relevant responses. Future iterations aim to support a broader range of file types and include an **AI Agent** for advanced query handling.

---

## Features
### Current Features
- **PDF Parsing**: Extracts information from PDF documents and generates responses using Hugging Face Open Source Model.
- **Contextual Query Handling**: Provides accurate answers based on the contents of uploaded PDFs.

### Future Enhancements
  - Other common document formats
- **AI Agent Integration**:
  - Enhanced conversational capabilities.
  - Ability to synthesize responses across multiple documents and file types.
- **Multi-Document Analysis**: Simultaneous querying across multiple files.
- **Search and Summarization**: Advanced document search and concise summaries for quick insights.

---

## Technical Details
- **Backend**: Hugging Face Open source for natural language understanding and generation.
- **Input Format**: PDF (current version), CSV, XLSX
- **Output**: Text-based responses tailored to user queries.
- **Frontend**: Stream-lit based UI

### Architecture
1. **File Upload Module**: Handles PDF uploads and validates file format.
2. **Preprocessing**: Extracts text from PDFs using OCR (if necessary) and prepares data for GPT-4.
3. **Query Engine**:
   - Matches user queries with relevant document content.
   - Generates responses using GPT-4.
4. **Response Module**: Returns precise and context-aware answers.

### Future Architecture Upgrades
- **File Format Conversion**: Incorporate libraries for handling diverse file types.
- **AI Agent Layer**: A conversational AI module capable of cross-referencing data and learning from interactions.

---

## Installation and Setup
### Prerequisites
- Python 3.8+
- Virtual Environment
- Required libraries (specified in `requirements.txt`):
  - `openai`
  - `PyPDF2`
  - `langchain`
  - `faiss-cpu`

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/nanomatter/RAG-Nanomatter.git
   ```
2. Navigate to the project directory:
   ```bash
   cd rag-app
   ```
3. Create and activate a virtual environment:
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the application:
   ```bash
   python app.py
   ```

---

## Usage
1. Launch the app and upload a PDF document.
2. Enter your query in the input field.
3. Receive a detailed response based on the document content.

---

## Roadmap
### Phase 1: Current Version
- Implement PDF parsing and query response using GPT-4.

### Phase 2: Multi-Format Support
- Add compatibility for CSV, TXT, XLSX, DOCX, and other formats.

### Phase 3: AI Agent Integration
- Develop an intelligent AI Agent for:
  - Advanced queries.
  - Multi-document handling.
  - Continuous learning.

### Phase 4: Advanced Features
- Implement robust search functionalities.
- Develop user-friendly dashboards for document management.

---

## License
This project is licensed under the Apache - 2.0.
