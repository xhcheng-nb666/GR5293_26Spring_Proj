# First-Pass Evaluation Summary

## Average Scores
- Baseline correctness: 2.00 / 2.00
- RAG correctness: 1.40 / 2.00
- RAG groundedness: 1.90 / 2.00
- Retrieval relevance: 1.40 / 2.00

## Strong Cases
- q1: What is RAG?
- q2: What is RLHF?
- q3: What is the bias-variance tradeoff?
- q5: What is the difference between semantic search and RAG?
- q6: What is benchmarking?
- q9: What is batching?

## Failure Cases
- q4: What is LoRA?
- q8: What are some challenges of training LLMs on consumer hardware?
- q10: What is the difference between prompt tuning and LoRA?

## Key Findings
- The baseline model is broadly knowledgeable on definition-style questions and achieved 20/20 first-pass correctness.
- The RAG system achieved lower correctness, 14/20, but much higher source discipline, with 19/20 groundedness.
- The main failure bottleneck is retrieval relevance, also 14/20, rather than answer generation itself.
- When retrieval succeeds, RAG makes course-grounding visible through retrieved lecture chunks and source pages.
- When retrieval fails, especially for LoRA and consumer-hardware questions, the RAG answer becomes incomplete even though the generation prompt behaves as intended.
- A top-k ablation suggests that k = 4 gives the best default balance between useful context and retrieval noise.
