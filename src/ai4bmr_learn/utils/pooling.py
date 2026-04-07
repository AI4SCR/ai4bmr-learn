def pool(x, strategy: str):
    if strategy is None:
        return x

    assert x.ndim == 3, f'Expected input with 3 dimensions, got {x.ndim} dimensions.'
    match strategy:
        case 'cls':
            return x[:, 0]
        case 'token':
            return x[:, 0]
        case 'flatten':
            return x.flatten(start_dim=1)
        case 'avg':
            return x.mean(dim=1)
        case "max":
            return x.amax(dim=1)
        case _:
            raise NotImplementedError(f'{strategy} is not implemented.')
