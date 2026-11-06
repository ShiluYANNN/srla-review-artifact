# SRLA

```bash
pip install -r requirements.txt
python src/evaluate.py --model /path/to/checkpoint --condition srla --format chat --device cuda:0 --seed 42 --output ../srla_outputs/evaluation
```

Use `--condition base` for the paired baseline. Inputs are in `data/prompts.jsonl`; evaluation settings are in `configs/evaluation.json`.

Seed construction:

```bash
python src/build_seeds.py --model /path/to/Qwen2.5-7B --device cuda:0 --seed 42 --ngram-size 6 --min-repeats 4 --output ../srla_outputs/prompts.jsonl
```
