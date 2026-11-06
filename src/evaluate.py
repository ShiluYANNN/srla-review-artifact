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
    import torch
    torch.manual_seed(seed)
    if device.startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
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



def generate_batch(model, tokenizer, encoded, seeds, config, device):

    import torch
    if len(seeds) != encoded.input_ids.shape[0]:
        raise ValueError('Each batch row requires a sampling seed')
    if tokenizer.pad_token_id is None:
        raise ValueError('A padding token is required')
    mask = encoded.attention_mask
    if (mask.sum(-1) == 0).any():
        raise ValueError('Empty tokenized prompt')
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
    generators = [torch.Generator(device=device).manual_seed(seed) for seed in seeds]
    ids = [[] for _ in seeds]
    steps = [[] for _ in seeds]
    stopped = [False] * len(seeds)
    positions = (mask.cumsum(-1)-1).clamp_min(0)
    with torch.inference_mode():
        output = model(input_ids=encoded.input_ids, attention_mask=mask,
                       position_ids=positions, use_cache=True)
        for step in range(config['max_new_tokens']):
            logits = output.logits[:, -1].float()
            probabilities = logits.softmax(-1)
            effective = torch.special.entr(probabilities).sum(-1).exp().tolist()
            eos_probabilities = probabilities[:, query].tolist()
            next_ids = []
            for i in range(len(seeds)):
                if stopped[i]:
                    next_ids.append(tokenizer.pad_token_id)
                    continue
                scaled = probabilities[i] if config['temperature'] == 1.0 else (logits[i]/config['temperature']).softmax(-1)
                distribution, indices, size = nucleus(scaled, config['top_p'])
                token = int(indices[torch.multinomial(distribution, 1, generator=generators[i])].item())
                next_ids.append(token)
                ids[i].append(token)
                steps[i].append({'p_selected': float(probabilities[i, token]),
                                 'k_eff': effective[i], 'p_eos': eos_probabilities[i],
                                 'nucleus_size': size})
                stopped[i] = token in eos
            if all(stopped) or step+1 == config['max_new_tokens']:
                break
            new_mask = mask.new_tensor([[0 if done else 1] for done in stopped])
            mask = torch.cat((mask, new_mask), dim=-1)
            positions = (mask.sum(-1, keepdim=True)-1).clamp_min(0)
            output = model(input_ids=encoded.input_ids.new_tensor(next_ids).unsqueeze(-1),
                           attention_mask=mask, position_ids=positions,
                           past_key_values=output.past_key_values, use_cache=True)
    return [{'token_ids': tokens, 'steps': stats, 'stopped': ended,
             'text': tokenizer.decode(tokens, skip_special_tokens=True),
             'metrics': summarize(tokens, stats, config['max_new_tokens'], ended)}
            for tokens, stats, ended in zip(ids, steps, stopped)]
