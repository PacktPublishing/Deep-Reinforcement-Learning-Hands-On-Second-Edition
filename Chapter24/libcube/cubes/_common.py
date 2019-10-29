def _permute(t, m, is_inv=False):
    """
    Perform permutation of tuple according to mapping m
    """
    r = list(t)
    for from_idx, to_idx in m:
        if is_inv:
            r[from_idx] = t[to_idx]
        else:
            r[to_idx] = t[from_idx]
    return r


def _rotate(corner_ort, corners):
    """
    Rotate given corners 120 degrees
    """
    r = list(corner_ort)
    for c, angle in corners:
        r[c] = (r[c] + angle) % 3
    return r


# orient corner cubelet
def _map_orient(cols, orient_id):
    if orient_id == 0:
        return cols
    elif orient_id == 1:
        return cols[2], cols[0], cols[1]
    else:
        return cols[1], cols[2], cols[0]

