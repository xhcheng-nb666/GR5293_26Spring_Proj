# Troubleshooting Guide

This guide lists common issues that may occur when setting up, running, or evaluating the RAG-Based Course Assistant.

## 1. `AGICTO_API_KEY` is missing

### Symptom

Generation, baseline, evaluation, or Streamlit demo scripts fail with an API key error.

### Cause

The current generation backend uses AGICTO's OpenAI-compatible API. The retrieval pipeline can run without an API key, but answer generation requires one.

### Fix

Set the API key in the same terminal session before running generation-related scripts:

```bash
export AGICTO_API_KEY="your_api_key_here"
```

Verify that the key is available:

```bash
echo $AGICTO_API_KEY
```

Then rerun the script, for example:

```bash
python src/rag_qa_agicto.py
```

or:

```bash
streamlit run app/streamlit_app.py
```

## 2. FAISS index or metadata file is missing

### Symptom

The retriever or Streamlit app cannot find one or both of the following files:

```text
data/index/faiss.index
data/index/chunk_metadata.json
```

### Cause

The index has not been built yet, or generated artifacts were not included in the local checkout.

### Fix

Run the pipeline from extraction to indexing:

```bash
python src/extract_pdf.py
python src/chunk_text.py
python src/build_index.py
```

After this, check that the following files exist:

```text
data/processed/all_pages.json
data/processed/all_chunks.json
data/index/faiss.index
data/index/chunk_metadata.json
```

Then test retrieval:

```bash
python src/test_retrieval.py
```

## 3. Tesseract OCR is not installed or not found

### Symptom

PDF extraction fails during OCR fallback, or image-heavy pages produce little or no text.

### Cause

Some lecture PDF pages may require OCR fallback. The extraction script expects Tesseract to be installed locally.

### Fix

Install Tesseract.

On macOS:

```bash
brew install tesseract
```

On Ubuntu / Debian:

```bash
sudo apt update
sudo apt install tesseract-ocr
```

Then rerun:

```bash
python src/extract_pdf.py
```

## 4. Retrieval results are weak or irrelevant

### Symptom

The RAG answer is incomplete, overly cautious, or says the retrieved context is insufficient.

### Cause

The top retrieved chunks may not contain the most relevant lecture material. This is a known limitation for some topics, especially acronym-heavy or concept-specific questions.

Known weaker cases include:

```text
What is LoRA?
What are some challenges of training LLMs on consumer hardware?
What is the difference between prompt tuning and LoRA?
```

### Fix

Try a larger retrieval depth in the Streamlit demo, or run the top-k ablation script:

```bash
python src/ablation_topk.py
```

For the current evaluation, `k=4` is used as the default because it gives the best balance between useful context and retrieval noise.

Potential future fixes include:

- hybrid BM25 + dense retrieval
- query expansion for acronyms
- reranking over top-k chunks
- OCR cleanup for noisy slide text
- further top-k and chunk-size sweeps

## 5. API rate limits or temporary backend failures

### Symptom

Batch evaluation fails partway through, or the backend returns temporary errors.

### Cause

External generation APIs may have rate limits, temporary availability issues, or request failures.

### Fix

The evaluation script saves partial progress. Rerun:

```bash
python src/evaluate_systems.py
```

Completed questions can be inspected in:

```text
eval/results_first_pass.json
```

If failures continue, wait briefly and rerun the same command. The retrieval-only parts of the project can still be inspected without an API key.

## 6. Streamlit app does not launch

### Symptom

The command below fails:

```bash
streamlit run app/streamlit_app.py
```

### Possible causes

- dependencies were not installed
- the virtual environment is not activated
- generated FAISS artifacts are missing
- the API key is not set for generation
- Streamlit is not installed in the active Python environment

### Fix

Activate the virtual environment and install dependencies:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

Rebuild the retrieval artifacts if needed:

```bash
python src/extract_pdf.py
python src/chunk_text.py
python src/build_index.py
```

Set the API key if you want generation to work:

```bash
export AGICTO_API_KEY="your_api_key_here"
```

Then rerun:

```bash
streamlit run app/streamlit_app.py
```

## 7. `pytest` is not available

### Symptom

Running tests fails with:

```bash
pytest: command not found
```

### Cause

`pytest` is not installed in the current Python environment.

### Fix

Install project dependencies:

```bash
pip install -r requirements.txt
```

If `pytest` is not included in the requirements file, install it directly:

```bash
pip install pytest
```

Then run the integration test:

```bash
pytest tests/test_retrieval_integration.py
```

## 8. Saved evaluation files are missing

### Symptom

The `eval/` folder does not contain expected outputs such as:

```text
eval/results_first_pass.json
eval/manual_scores.csv
eval/results_summary.md
eval/topk_ablation.md
```

### Cause

The evaluation or ablation scripts may not have been run yet.

### Fix

Run:

```bash
python src/evaluate_systems.py
python src/ablation_topk.py
python src/summarize_scores.py
```

Then inspect the generated files under `eval/`.

## 9. Results differ slightly across runs

### Symptom

Generated answers are not exactly identical when rerunning baseline or RAG generation.

### Cause

The retrieval pipeline is deterministic after the index is built, but final answer generation depends on an external LLM API. Small wording differences may occur across runs.

### Fix

For reproducibility, use the saved first-pass outputs in:

```text
eval/results_first_pass.json
eval/manual_scores.csv
eval/results_summary.md
```

When reporting results, clearly distinguish between:

- reproducible retrieval artifacts
- saved evaluation outputs
- generation outputs that may vary slightly across API calls
