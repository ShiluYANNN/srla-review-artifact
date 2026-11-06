import argparse
import json
from pathlib import Path

from evaluate import render


def repeated_span(token_ids, ngram_size, min_repeats):
    if len(token_ids) < ngram_size:
        return None
    pattern = token_ids[-ngram_size:]
    positions = [i for i in range(len(token_ids) - ngram_size + 1)
                 if token_ids[i:i + ngram_size] == pattern]
    if len(positions) < min_repeats:
        return None
    return token_ids[positions[0]:]


def construct_seed(model, tokenizer, text, seed, style, device, ngram_size, min_repeats):
    import torch

    eos = set()
    for value in (model.generation_config.eos_token_id,
                  model.config.eos_token_id, tokenizer.eos_token_id):
        if isinstance(value, int):
            eos.add(value)
        elif value is not None:
            eos.update(value)
    if not eos:
        raise ValueError('An EOS token is required')
    encoded = tokenizer(render(tokenizer, text, style), return_tensors='pt',
                        add_special_tokens=False).to(device)
    if encoded.input_ids.shape[1] == 0:
        raise ValueError('Empty input')
    context = getattr(model.config, 'max_position_embeddings', None)
    if context and encoded.input_ids.shape[1] + 512 > context:
        raise ValueError('Input and continuation exceed model context capacity')
    generator = torch.Generator(device=device).manual_seed(seed)
    ids = []
    mask = encoded.attention_mask
    with torch.inference_mode():
        output = model(**encoded, use_cache=True)
        for step in range(512):
            logits = output.logits[0, -1].float().clone()
            logits[list(eos)] = -torch.inf
            distribution = (logits / 0.5).softmax(-1)
            token = int(torch.multinomial(distribution, 1, generator=generator))
            ids.append(token)
            span = repeated_span(ids, ngram_size, min_repeats)
            if span is not None:
                return tokenizer.decode(span, skip_special_tokens=False)
            if step == 511:
                break
            mask = torch.cat((mask, mask.new_ones((1, 1))), dim=-1)
            output = model(input_ids=encoded.input_ids.new_tensor([[token]]),
                           attention_mask=mask, past_key_values=output.past_key_values,
                           use_cache=True)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen2.5-7B')
    parser.add_argument('--data', default='data/prompts.jsonl')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--format', choices=['raw', 'chat'], default='raw')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--ngram-size', type=int, required=True)
    parser.add_argument('--min-repeats', type=int, required=True)
    args = parser.parse_args()
    if args.ngram_size < 1 or args.min_repeats < 2:
        parser.error('Invalid repetition rule')
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    if not rows or len({row['id'] for row in rows}) != len(rows):
        parser.error('Input IDs must be unique and nonempty')
    if any(not isinstance(row.get('base'), str) or not row['base'].strip() for row in rows):
        parser.error('Every input requires a nonempty base')
    output_path = Path(args.output)
    if output_path.exists():
        parser.error('Use a new output file')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32 if args.device == 'cpu' else torch.float16
    ).to(args.device).eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('x', encoding='utf-8') as output:
        for index, row in enumerate(rows):
            suffix = construct_seed(model, tokenizer, row['base'], args.seed + index,
                                    args.format, args.device, args.ngram_size, args.min_repeats)
            if suffix is not None:
                result = {'id': row['id'], 'language': row['language'],
                          'base': row['base'], 'srla': row['base'] + '\n\n' + suffix}
                output.write(json.dumps(result, ensure_ascii=False) + '\n')
                output.flush()


if __name__ == '__main__':
    main()
