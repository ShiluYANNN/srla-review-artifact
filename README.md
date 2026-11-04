# SRLA Evaluation

Implementation and 332 paired prompts: 82 Chinese and 250 English. Each record contains `base` and `srla`. The SRLA input preserves the question and appends its continuation.

Install `requirements.txt` with a PyTorch build compatible with your CUDA environment, then run:

```bash
python src/evaluate.py --model /path/to/checkpoint --condition srla --format chat --device cuda:0 --seed 42 --output runs/eval
```

Use `--condition base` for the paired baseline, or `--format raw` for models without a chat template. Increase `--batch-size` if GPU memory permits.

`configs/evaluation.json` specifies random sampling, temperature 1.0, top-p 1.0, top-k disabled, a 4096-token budget and three generations per prompt (996 total). No repetition penalty is applied.

The evaluator writes generation records, token metrics, RDR and a run manifest. RDR requires budget exhaustion and a 6-token pattern appearing at least four times in the final quarter. Native-distribution entropy and sampling-support size are computed separately.
