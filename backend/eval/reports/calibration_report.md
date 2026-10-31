# DocuGround Threshold Calibration Report

## Recommended Thresholds
| Parameter | Value |
|---|---|
| **DENSE_SCORE_CUTOFF** | `0.1` |
| **RERANKER_SCORE_CUTOFF** | `0.25` |
| **Correct Refusal Rate** | `40.0%` (Target: >= 95%) |
| **False Refusal Rate** | `2.9%` (Target: <= 5%) |

## Score Distributions
### Dense Cosine Similarity
- **Answerable Questions**: Mean=0.771, Median=0.770, Range=[0.759, 0.784]
- **Unanswerable Questions**: Mean=0.768, Median=0.767, Range=[0.759, 0.783]

### Reranker Cross-Encoder Score
- **Answerable Questions**: Mean=0.410, Median=0.385, Range=[0.231, 0.667]
- **Unanswerable Questions**: Mean=0.278, Median=0.250, Range=[0.133, 0.600]
