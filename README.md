# GR5293 Final Project: RAG-Based Course Assistant

This project builds an end-to-end Retrieval-Augmented Generation (RAG) assistant for a Generative AI course.  
The system answers course-related questions using lecture slides as its document base, instead of relying only on the LLM's prior knowledge.

## Project Goal

The goal is to evaluate whether a course-specific RAG system can produce answers that are more grounded in course materials than a plain LLM baseline.

Core comparison:
- **Baseline:** Gemini answers the question directly
- **RAG system:** retrieve relevant lecture chunks first, then answer using only those retrieved chunks

## Current Pipeline

1. **Document ingestion**
   - Lecture PDFs are stored in `data/raw/`
   - PDFs are parsed with PyMuPDF
   - If native extraction fails on a page, OCR fallback is used with Tesseract

2. **Preprocessing**
   - Extracted pages are stored in structured JSON
   - Pages are lightly cleaned
   - Pages are chunked into retrieval-ready chunks

3. **Retrieval**
   - Chunks are embedded with `sentence-transformers/all-MiniLM-L6-v2`
   - Embeddings are stored in a FAISS index
   - Top-k chunks are retrieved for each query

4. **Generation**
   - Retrieved chunks are inserted into a grounded prompt
   - Gemini generates the final answer using only the retrieved material

5. **Evaluation**
   - Questions are run through both the baseline and the RAG system
   - Outputs are saved for manual comparison
   - Manual scoring focuses on correctness, groundedness, and retrieval relevance

## Repository Structure

```text
GR5293_Proj/
  app/
    streamlit_app.py
  data/
    raw/
    processed/
    index/
  eval/
    questions.json
    results_first_pass.json
    manual_scores.csv
  src/
    extract_pdf.py
    chunk_text.py
    build_index.py
    test_retrieval.py
    baseline_gemini.py
    rag_qa_gemini.py
    evaluate_systems.py
  report/
  slides/
  requirements.txt
  README.md
  .gitignore
```

## Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Tesseract OCR

On macOS:

```bash
brew install tesseract
```

On Ubuntu / Debian:

```bash
sudo apt update
sudo apt install tesseract-ocr
```

## API Key Setup

This project uses Gemini for generation.

Set your Gemini API key in the same terminal session before running the scripts:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

You can verify it with:

```bash
echo $GEMINI_API_KEY
```

## How to Run the Pipeline

### Step 1: Extract PDFs

```bash
python src/extract_pdf.py
```

This:
- reads PDFs from `data/raw/`
- uses native text extraction first
- falls back to OCR for low-text pages
- saves page-level JSON and TXT outputs

### Step 2: Chunk the extracted pages

```bash
python src/chunk_text.py
```

This produces `data/processed/all_chunks.json`.

### Step 3: Build the retrieval index

```bash
python src/build_index.py
```

This creates:
- `data/index/faiss.index`
- `data/index/chunk_metadata.json`

### Step 4: Test retrieval

```bash
python src/test_retrieval.py
```

Use this to inspect whether the retriever returns relevant chunks for sample course questions.

### Step 5: Run the baseline

```bash
python src/baseline_gemini.py
```

This asks Gemini the question directly, without retrieval.

### Step 6: Run the RAG assistant

```bash
python src/rag_qa_gemini.py
```

This:
- retrieves top-k chunks from FAISS
- builds a grounded prompt
- generates an answer using Gemini
- prints retrieved sources

### Step 7: Run evaluation

```bash
python src/evaluate_systems.py
```

This script compares:
- baseline Gemini answers
- RAG answers

It also:
- saves partial progress
- resumes from completed questions
- includes retry logic for Gemini free-tier rate limits

## Evaluation Design

The first-pass evaluation uses course-related questions such as:
- What is RAG?
- What is RLHF?
- What is the bias-variance tradeoff?
- What is LoRA?
- What is the difference between semantic search and RAG?

Manual scoring fields:
- `baseline_correctness`
- `rag_correctness`
- `rag_groundedness`
- `retrieval_relevance`

## Current Findings

Early results suggest:
- the baseline Gemini model is often broadly correct on definition-style questions
- the RAG system is more explicitly grounded in course materials when retrieval succeeds
- the main bottleneck is retrieval quality, not generation quality
- failure cases occur when the retriever misses the best lecture chunk

## Demo Plan

The intended demo flow is:
1. user asks a course question
2. system retrieves top lecture chunks
3. system generates a grounded answer
4. system displays source lecture pages

## Future Improvements

Planned next improvements include:
- improving retrieval for failure cases such as LoRA and consumer-hardware questions
- adding a simple Streamlit interface
- completing q9/q10 evaluation examples
- running one small retrieval ablation such as top-k or chunk-size tuning

## Notes

- Large generated files under `data/processed/` and `data/index/` should usually not be committed.
- Raw lecture PDFs may also be excluded from Git if they are large or course-restricted.
- The first working goal is a stable, reproducible, end-to-end RAG assistant before optimization.
