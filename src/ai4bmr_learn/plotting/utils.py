
def get_colorcet_map(item: list, as_int: bool = True) -> dict:
    import colorcet as cc
    uniq = sorted(set(item))
    glasbey_colors = cc.glasbey_bw[:len(uniq)]
    scale = 255 if as_int else 1
    color_map = {
        i: tuple(int(scale * c) for c in rgb)
        for i, rgb in zip(uniq, glasbey_colors)
    }
    return color_map
