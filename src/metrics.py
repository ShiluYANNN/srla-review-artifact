from collections import Counter
import math


def repetition(tokens, fraction=0.25, ngram_size=6, min_repeats=4):
    if not 0 < fraction <= 1 or ngram_size < 1 or min_repeats < 2:
        raise ValueError('Invalid repetition rule')
    tail = tokens[math.floor((1-fraction)*len(tokens)):]
    counts = Counter(tuple(tail[i:i+ngram_size]) for i in range(len(tail)-ngram_size+1))
    return max(counts.values(), default=0) >= min_repeats


def summarize(token_ids, steps, max_new_tokens, stopped):
    if not token_ids or len(token_ids) != len(steps):
        raise ValueError('Missing or inconsistent token records')
    tail = steps[3*len(steps)//4:]
    return {
        'tokens': len(token_ids),
        'rdr': len(token_ids) == max_new_tokens and not stopped and repetition(token_ids),
        'q99': sum(s['p_selected'] >= .99 for s in steps)/len(steps),
        'k_eff_late': sum(s['k_eff'] for s in tail)/len(tail),
        'p_eos_late': sum(s['p_eos'] for s in tail)/len(tail),
        'nucleus_size_late': sum(s['nucleus_size'] for s in tail)/len(tail),
    }
