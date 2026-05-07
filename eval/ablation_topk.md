| Question                                                | k=2             | k=4    | k=6                 | Best |
| ------------------------------------------------------- | --------------- | ------ | ------------------- | ---- |
| What is RAG?                                            | good            | strong | slightly noisy      | 4    |
| What is RLHF?                                           | good            | strong | similar but noisier | 4    |
| What is the bias-variance tradeoff?                     | weak/incomplete | strong | strong but verbose  | 4    |
| What is LoRA?                                           | fail            | fail   | improves            | 6    |
| What is the difference between semantic search and RAG? | strong          | strong | similar             | 4    |

## Conclusion
In the top-k ablation, k=4 provided the best overall tradeoff between answer quality and retrieval noise. Smaller retrieval depth (k=2) sometimes omitted necessary supporting context, as seen in the bias-variance question, while larger retrieval depth (k=6) occasionally improved difficult cases such as LoRA but also introduced more irrelevant chunks. This suggests that moderate retrieval depth is a good default, though certain harder queries may benefit from deeper retrieval.