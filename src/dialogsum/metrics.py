import numpy as np
import sacrebleu
from rouge_score import rouge_scorer

ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]


def compute_metrics(predictions, references):
    scorer = rouge_scorer.RougeScorer(ROUGE_TYPES, use_stemmer=True)
    per_type = {t: [] for t in ROUGE_TYPES}
    for pred, refs in zip(predictions, references):
        scores = scorer.score_multi(refs, pred)
        for t in ROUGE_TYPES:
            per_type[t].append(scores[t].fmeasure)
    result = {t: float(np.mean(v)) * 100 for t, v in per_type.items()}

    n_refs = max(len(r) for r in references)
    ref_streams = [[refs[i] if i < len(refs) else refs[0] for refs in references] for i in range(n_refs)]
    result["bleu"] = float(sacrebleu.corpus_bleu(predictions, ref_streams).score)
    result["pred_len_words"] = float(np.mean([len(p.split()) for p in predictions]))
    return result
