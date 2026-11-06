from .inputs import prepare_inputs


def repeated_span(token_ids, ngram_size, min_repeats):
    if len(token_ids) < ngram_size:
        return None
    pattern = token_ids[-ngram_size:]
    positions = [i for i in range(len(token_ids) - ngram_size + 1)
                 if token_ids[i:i + ngram_size] == pattern]
    if len(positions) < min_repeats:
        return None
    return token_ids[positions[0]:]


def construct_seed(model, tokenizer, encoded, seed, device, ngram_size, min_repeats):
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
    if ngram_size < 1 or min_repeats < 2:
        raise ValueError('Invalid repetition rule')
    input_ids, mask = prepare_inputs(encoded, model.config, 512, 1)
    generator = torch.Generator(device=device).manual_seed(seed)
    ids = []
    with torch.inference_mode():
        output = model(input_ids=input_ids, attention_mask=mask, use_cache=True)
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
            output = model(input_ids=input_ids.new_tensor([[token]]),
                           attention_mask=mask, past_key_values=output.past_key_values,
                           use_cache=True)
    return None
