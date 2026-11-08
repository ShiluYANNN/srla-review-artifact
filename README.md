# SRLA

Seed construction and evaluation functions take tokenized model inputs. Paired prompts are in `data/prompts.jsonl`.

The full prompt and generation budget must fit the configured context window. RDR counts budget exhaustion with a repeated 6-gram occurring at least four times in the final quarter. `p_eos` tracks the tokenizer EOS token. `p_stop_raw` and `p_stop_sampled` sum the enabled stop-token probabilities before and after sampling transforms.

Tail prompts are available in [JSONL format](data/tail142/prompts.jsonl).
