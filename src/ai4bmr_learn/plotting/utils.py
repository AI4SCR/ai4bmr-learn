def get_colorcet_map(items: list, as_int: bool = True) -> dict:
    import colorcet as cc

    unique_items = sorted(set(items))
    glasbey_colors = cc.glasbey_bw[: len(unique_items)]
    scale = 255 if as_int else 1
    return {
        item: tuple(int(scale * channel) for channel in rgb)
        for item, rgb in zip(unique_items, glasbey_colors)
    }
