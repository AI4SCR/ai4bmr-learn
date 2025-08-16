def pool(x, strategy: str):
    if strategy is None:
        return x
    elif strategy == 'cls':
        return x[:, 0]
    elif strategy == 'flatten':
        return x.flatten(start_dim=1)
    else:
        raise NotImplementedError(f'{strategy} is not implemented.')