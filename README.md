# SRLA

Seed construction and evaluation functions take tokenized model inputs. Paired prompts are in `data/prompts.jsonl`.

The full prompt and generation budget must fit the configured context window. RDR counts budget exhaustion with a repeated 6-gram occurring at least four times in the final quarter. `p_eos` tracks the tokenizer EOS token. `p_stop_raw` and `p_stop_sampled` sum the enabled stop-token probabilities before and after sampling transforms.

The [Tail comparison subset](data/tail142/README.md) provides the 142 generation keys, question and Tail texts, and per-model outcomes.
