def prepare_inputs(encoded, config, budget, batch_size):
    import torch

    ids = encoded.input_ids
    mask = encoded.attention_mask
    if ids.ndim != 2 or ids.shape != mask.shape or not batch_size or ids.size(0) != batch_size:
        raise ValueError('Expected one attention mask and one input per sampling seed')
    if not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError('Expected a binary attention mask')
    lengths = mask.long().sum(-1)
    if int(lengths.min()) == 0 or type(budget) is not int or budget < 1:
        raise ValueError('Prompt and generation budget must be nonempty')
    width = int(lengths.max())
    limit = getattr(config, 'max_position_embeddings', None)
    if not isinstance(limit, int) or limit < 1:
        raise ValueError('Model must specify its context capacity')
    if width + budget > limit:
        raise ValueError(f'Prompt ({width}) + generation budget ({budget}) exceeds context ({limit})')
    positions = torch.arange(width, device=ids.device)
    left_mask = positions >= width - lengths[:, None]
    valid = mask.bool()
    left_ids = ids.new_zeros((batch_size, width))
    left_ids[left_mask] = ids[valid]
    return left_ids, left_mask.long()
