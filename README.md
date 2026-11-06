# SRLA Evaluation

Evaluation of supplied paired multiple-choice prompts under native model decoding. Each input contains `id`, `language`, `base`, and `srla`; the SRLA input preserves the base question and appends its continuation. The dataset contains 330 unique pairs: 80 Chinese and 250 English.

Install `requirements.txt` with a compatible PyTorch build. From the repository root:

```bash
python src/evaluate.py --model /path/to/checkpoint --condition srla --format chat --device cuda:0 --seed 42 --output ../srla_outputs/model_srla
```

Use `--condition base` for the paired baseline. `--format chat` applies the checkpoint tokenizer's chat template; `--format raw` sends the supplied text directly. `--batch-size` controls execution batching.

The configuration uses temperature 0.7, top-p 0.9, no top-k, a 4096-token output budget, and three generations per prompt. Generation does not suppress EOS or apply a repetition penalty.

RDR requires budget exhaustion without a stop token and a 6-token pattern occurring at least four times in the final quarter of the generated token IDs. The evaluator computes selected-token confidence at threshold 0.99 over the full trajectory and native-distribution entropy-based effective support and tokenizer-EOS probability over the final quarter. Nucleus size is recorded separately from entropy-based effective support. The stopping set also includes checkpoint generation EOS identifiers and verified conversation-end tokens.

This package evaluates supplied prompts. Seed extraction, training, and benchmark capability evaluation are outside this package.
