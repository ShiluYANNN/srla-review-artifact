"""Evaluate paired prompts with explicit sampling and native token statistics."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import math
import re
import sys
from metrics import summarize


def render(tokenizer, text, style):
    if style == 'raw':
        return text
    messages = []
    for line in text.strip().splitlines():
        match = re.match(r'^(Human|Assistant):\s*(.*)', line)
        if match:
            messages.append({'role': 'user' if match[1] == 'Human' else 'assistant', 'content': match[2]})
        elif messages:
            messages[-1]['content'] += '\n' + line
        else:
            messages.append({'role': 'user', 'content': line})
    if not messages or not tokenizer.chat_template:
        raise ValueError('Chat mode requires a nonempty prompt and an official chat template')
    continuing = messages[-1]['role'] == 'assistant'
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=not continuing,
                                        continue_final_message=continuing)


def nucleus(probs, top_p):
    """Keep the smallest prefix reaching top_p, including its crossing token."""
    import torch
    if top_p == 1.0:
        return probs, torch.arange(probs.numel(), device=probs.device), probs.numel()
    values, indices = probs.sort(descending=True)
    keep = (values.cumsum(-1) - values) < top_p
    keep[0] = True
    weights = values * keep
    return weights / weights.sum(), indices, int(keep.sum())


def generate(model, tokenizer, text, config, seed, style, device):
    import torch
    torch.manual_seed(seed)
    if device.startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
    encoded = tokenizer(render(tokenizer, text, style), return_tensors='pt', add_special_tokens=False).to(device)
    if encoded.input_ids.shape[1] == 0:
        raise ValueError('Empty tokenized prompt')
    eos = set()
    for source in (model.generation_config.eos_token_id, tokenizer.eos_token_id):
        if isinstance(source, int): eos.add(source)
        elif source: eos.update(source)
    for marker in ('<|im_end|>', '<|eot_id|>'):
        idx = tokenizer.convert_tokens_to_ids(marker)
        if isinstance(idx, int) and tokenizer.convert_ids_to_tokens(idx) == marker:
            eos.add(idx)
    if not eos:
        raise ValueError('No EOS token identified')
    eos_query_id = tokenizer.eos_token_id
    if not isinstance(eos_query_id, int):
        raise ValueError('A single tokenizer EOS ID is required for the EOS probability metric')
    ids, steps, stopped = [], [], False
    with torch.inference_mode():
        output = model(**encoded, use_cache=True)
        mask = encoded.attention_mask
        for _ in range(config['max_new_tokens']):
            logits = output.logits[0, -1].float()
            p = logits.softmax(-1)
            distribution, indices, size = nucleus((logits/config['temperature']).softmax(-1), config['top_p'])
            token_id = int(indices[torch.multinomial(distribution, 1)].item())
            ids.append(token_id)
            steps.append({'p_selected': float(p[token_id]),
                          'k_eff': float(torch.special.entr(p).sum().exp()),
                          'p_eos': float(p[eos_query_id]), 'nucleus_size': size})
            if token_id in eos:
                stopped = True
                break
            if len(ids) == config['max_new_tokens']:
                break
            mask = torch.cat((mask, mask.new_ones((1,1))), dim=-1)
            output = model(input_ids=encoded.input_ids.new_tensor([[token_id]]), attention_mask=mask,
                           past_key_values=output.past_key_values, use_cache=True)
    return {'token_ids': ids, 'steps': steps, 'stopped': stopped,
            'text': tokenizer.decode(ids, skip_special_tokens=True),
            'metrics': summarize(ids, steps, config['max_new_tokens'], stopped)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--data', default='data/prompts.jsonl')
    ap.add_argument('--condition', choices=['base','srla'], required=True)
    ap.add_argument('--config', default='configs/evaluation.json')
    ap.add_argument('--output', required=True)
    ap.add_argument('--device', required=True, help='Explicit device, e.g. cuda:0')
    ap.add_argument('--format', choices=['raw','chat'], required=True)
    ap.add_argument('--seed', type=int, required=True)
    args = ap.parse_args()
    config = json.loads(Path(args.config).read_text())
    if set(config) - {'do_sample', 'temperature', 'top_p', 'top_k', 'max_new_tokens', 'repeats'}:
        ap.error('Unknown sampling configuration keys')
    if config.get('do_sample', True) is not True:
        ap.error('This evaluation requires random sampling')
    if config.get('top_k', 0) != 0:
        ap.error('This protocol uses top-p sampling without top-k; top_k must be 0')
    if not math.isfinite(config['temperature']) or not math.isfinite(config['top_p']) or not 0 < config['top_p'] <= 1 or config['temperature'] <= 0 or any(
            type(config[k]) is not int or config[k] < 1 for k in ('max_new_tokens','repeats')):
        ap.error('Invalid sampling configuration')
    data = [json.loads(s) for s in Path(args.data).read_text().splitlines() if s.strip()]
    if not data or len({row['id'] for row in data}) != len(data):
        ap.error('Dataset must be nonempty with unique record IDs')
    if any(not isinstance(row.get(args.condition), str) or not row[args.condition].strip() for row in data):
        ap.error('Selected condition must contain nonempty prompt text for every record')
    out = Path(args.output)
    if out.exists():
        ap.error('Use a new output directory')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32 if args.device=='cpu' else torch.float16).to(args.device).eval()
    out.mkdir(parents=True)
    manifest = {'arguments': vars(args), 'config': config, 'python': platform.python_version(),
                'versions': {k:importlib.metadata.version(k) for k in ('torch','transformers')},
                'dataset_sha256': hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                'expected_records': len(data)*config['repeats'],
                'checkpoint_commit': getattr(model.config, '_commit_hash', None),
                'seed_rule':'seed + row_index * repeats + repeat_index',
                'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    totals = []
    with (out/'generations.jsonl').open('x') as f:
        for index, row in enumerate(data):
            for repeat in range(config['repeats']):
                seed = args.seed + index*config['repeats'] + repeat
                result = generate(model, tokenizer, row[args.condition], config, seed, args.format, args.device)
                result.update(id=row['id'], repeat=repeat, seed=seed)
                f.write(json.dumps(result,ensure_ascii=False)+'\n'); f.flush()
                totals.append(result['metrics'])
    summary = {k:sum(r[k] for r in totals)/len(totals) for k in ('rdr','q99','k_eff_late','p_eos_late','nucleus_size_late')}
    summary.update(valid_generations=len(totals), prompts=len(data), rdr_percent=100*summary['rdr'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
