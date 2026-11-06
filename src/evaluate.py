from .inputs import prepare_inputs
from .metrics import summarize


def nucleus(probs, top_p):

    import torch
    if top_p == 1.0:
        return probs, torch.arange(probs.numel(), device=probs.device), probs.numel()
    values, indices = probs.sort(descending=True)
    keep = (values.cumsum(-1) - values) < top_p
    keep[0] = True
    weights = values * keep
    return weights / weights.sum(), indices, int(keep.sum())


def generate(model, tokenizer, encoded, config, seed, device):
    return generate_batch(model, tokenizer, encoded, [seed], config, device)[0]


def generate_batch(model, tokenizer, encoded, seeds, config, device):

    import torch
    budget = config['max_new_tokens']
    input_ids, mask = prepare_inputs(encoded, model.config, budget, len(seeds))
    temperature, top_p = config['temperature'], config['top_p']
    if not 0 < temperature < float('inf') or not 0 < top_p <= 1:
        raise ValueError('Invalid temperature or top_p')
    if config.get('do_sample', True) is not True or config.get('top_k', 0) != 0:
        raise ValueError('Expected sampling without top-k filtering')
    eos = set()
    for source in (model.generation_config.eos_token_id, tokenizer.eos_token_id):
        if isinstance(source, int): eos.add(source)
        elif source: eos.update(source)
    for marker in ('<|im_end|>', '<|eot_id|>'):
        token = tokenizer.convert_tokens_to_ids(marker)
        if isinstance(token, int) and tokenizer.convert_ids_to_tokens(token) == marker:
            eos.add(token)
    query = tokenizer.eos_token_id
    if not eos or not isinstance(query, int):
        raise ValueError('EOS configuration is required')
    stop_ids = sorted(eos)
    generators = [torch.Generator(device=device).manual_seed(seed) for seed in seeds]
    ids = [[] for _ in seeds]
    steps = [[] for _ in seeds]
    stopped = [False] * len(seeds)
    positions = (mask.cumsum(-1)-1).clamp_min(0)
    with torch.inference_mode():
        output = model(input_ids=input_ids, attention_mask=mask,
                       position_ids=positions, use_cache=True)
        for step in range(config['max_new_tokens']):
            logits = output.logits[:, -1].float()
            probabilities = logits.softmax(-1)
            effective = torch.special.entr(probabilities).sum(-1).exp().tolist()
            eos_probabilities = probabilities[:, query].tolist()
            raw_stop = probabilities[:, stop_ids].sum(-1).tolist()
            next_ids = []
            for i in range(len(seeds)):
                if stopped[i]:
                    next_ids.append(query)
                    continue
                scaled = probabilities[i] if config['temperature'] == 1.0 else (logits[i]/config['temperature']).softmax(-1)
                distribution, indices, size = nucleus(scaled, config['top_p'])
                token = int(indices[torch.multinomial(distribution, 1, generator=generators[i])].item())
                stop_mask = torch.isin(indices, indices.new_tensor(stop_ids))
                next_ids.append(token)
                ids[i].append(token)
                steps[i].append({'p_selected': float(probabilities[i, token]),
                                 'k_eff': effective[i], 'p_eos': eos_probabilities[i],
                                 'p_stop_raw': raw_stop[i],
                                 'p_stop_sampled': float(distribution[stop_mask].sum()),
                                 'nucleus_size': size})
                stopped[i] = token in eos
            if all(stopped) or step+1 == config['max_new_tokens']:
                break
            new_mask = mask.new_tensor([[0 if done else 1] for done in stopped])
            mask = torch.cat((mask, new_mask), dim=-1)
            positions = (mask.sum(-1, keepdim=True)-1).clamp_min(0)
            output = model(input_ids=input_ids.new_tensor(next_ids).unsqueeze(-1),
                           attention_mask=mask, position_ids=positions,
                           past_key_values=output.past_key_values, use_cache=True)
    return [{'token_ids': tokens, 'steps': stats, 'stopped': ended,
             'text': tokenizer.decode(tokens, skip_special_tokens=True),
             'metrics': summarize(tokens, stats, config['max_new_tokens'], ended)}
            for tokens, stats, ended in zip(ids, steps, stopped)]
