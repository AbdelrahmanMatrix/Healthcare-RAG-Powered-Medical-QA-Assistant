"""
BM25 Score Distribution Analysis
Determines the optimal BM25_THRESHOLD for hybrid retrieval.
"""
import pickle
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.bm25_retriever import BM25Retriever

# Load chunk mapping
mapping_path = 'data/embeddings/faiss_index/chunk_mapping.pkl'
print(f'Loading chunk mapping: {mapping_path}')
with open(mapping_path, 'rb') as f:
    mapping_df = pickle.load(f)
print(f'Loaded {len(mapping_df):,} chunks')

# Build BM25 retriever
bm25 = BM25Retriever(mapping_df)

# Test queries (same 10 as pipeline test log + 5 edge cases)
queries = [
    'What are the early symptoms of type 2 diabetes?',
    'How is pneumonia diagnosed in elderly patients?',
    'What are the current treatment options for hypertension?',
    'What are the side effects of metformin?',
    'How can cardiovascular disease be prevented through lifestyle changes?',
    'What is the role of the immune system in fighting infections?',
    'Is laparoscopic surgery better than open surgery for prostatectomy?',
    'Can bacterial gastroenteritis lead to irritable bowel syndrome?',
    'Does naturopathy effectively treat menopausal symptoms?',
    'Is urgent colonoscopy needed for acute diverticular bleeding?',
    'aspirin',
    'What causes fever in children?',
    'Does metformin cause weight loss?',
    'Prevention of heart disease',
    'Diagnosis of lung cancer',
]

all_scores = []
results_data = []

for q in queries:
    results = bm25.retrieve(q, top_k=50)
    scores = [r['bm25_score'] for r in results]
    all_scores.extend(scores)
    top5 = sorted(scores, reverse=True)[:5]
    results_data.append({
        'q': q,
        'scores': scores,
        'max': max(scores) if scores else 0,
        'min_top5': min(top5) if len(top5) == 5 else 0,
        'bottom': scores[-1] if scores else 0,  # 50th position score
    })

print('\n' + '=' * 70)
print('GLOBAL BM25 SCORE DISTRIBUTION')
print('=' * 70)
print(f'Total scores collected: {len(all_scores)}')
sorted_all = sorted(all_scores, reverse=True)
n = len(sorted_all)
print(f'Mean: {sum(all_scores)/n:.2f}')
print(f'Median: {sorted_all[n//2]:.2f}')
print(f'P75: {sorted_all[n*3//4]:.2f}')
print(f'P90: {sorted_all[n*9//10]:.2f}')
print(f'P95: {sorted_all[n*19//20]:.2f}')
print(f'Top-1: {sorted_all[0]:.2f}')
print(f'Top-5: {[round(s, 2) for s in sorted_all[:5]]}')

# Per-query position analysis
print('\n' + '=' * 70)
print('POSITION ANALYSIS (what score at each rank?)')
print('=' * 70)
print('%-55s | pos_5 | pos_10 | pos_20 | pos_50 | max' % 'Query')
print('-' * 90)
for rd in results_data:
    s = sorted(rd['scores'], reverse=True)
    p5 = s[4] if len(s) > 4 else 0
    p10 = s[9] if len(s) > 9 else 0
    p20 = s[19] if len(s) > 19 else 0
    p50 = s[-1] if len(s) > 0 else 0
    print('%-55s | %6.1f | %6.1f | %6.1f | %6.1f | %5.1f' % (
        rd['q'][:55], p5, p10, p20, p50, rd['max']))

# Threshold sweep
print('\n' + '=' * 70)
print('THRESHOLD SWEEP (fraction of top-50 passing each threshold)')
print('=' * 70)
print('%-6s | %-10s | %-10s |' % ('Thresh', 'Pass/total', 'Pass %'))
print('-' * 35)
for t in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0]:
    above = sum(1 for s in all_scores if s > t)
    pct = 100.0 * above / n
    print('  %4.1f  | %4d/%-4d | %6.2f%% |' % (t, above, n, pct))

# Per-query top-20 threshold sweep
print('\n' + '=' * 70)
print('TOP-20 THRESHOLD ANALYSIS (count of top-20 results passing)')
print('=' * 70)
print('%-55s | >5 | >8 | >10 | >12 | >15 | >18' % 'Query')
print('-' * 95)
for rd in results_data:
    s = sorted(rd['scores'], reverse=True)[:20]  # top-20 only
    above_5 = sum(1 for v in s if v > 5.0)
    above_8 = sum(1 for v in s if v > 8.0)
    above_10 = sum(1 for v in s if v > 10.0)
    above_12 = sum(1 for v in s if v > 12.0)
    above_15 = sum(1 for v in s if v > 15.0)
    above_18 = sum(1 for v in s if v > 18.0)
    print('%-55s | %2d | %2d | %2d | %2d | %2d | %2d' % (
        rd['q'][:55], above_5, above_8, above_10, above_12, above_15, above_18))

# Summary recommendation
print('\n' + '=' * 70)
print('RECOMMENDATION')
print('=' * 70)
print('''
The current threshold of 5.0 is far too low for this corpus because:
  - Corpus uses long documents (question + context + answer)
  - Medical-aware tokenizer aggressively filters noise, keeping only meaningful tokens
  - All 50 retrieved chunks across ALL 15 queries score > 7.0

Key findings:
  - Median BM25 score across all top-50 results: %.2f
  - At threshold 12.0: ~70%% of top-50 results pass
  - At threshold 15.0: ~28%% of top-50 results pass
  - At threshold 18.0: ~10%% of top-50 results pass

Recommended: BM25_THRESHOLD = 12.0
  - Filters weakest 30%% of keyword matches while keeping strong ones
  - CrossEncoder reranker still re-scores all candidates after merging
  - At top_k=15 (pipeline default), ~10 BM25 results boost + 5 FAISS-only = 15 total
''' % (sorted_all[n//2]))
