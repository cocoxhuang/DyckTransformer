def word_to_path(tokens):
    '''Convert a Dyck word (0/1 steps) into lattice path coordinates.

    Interprets tokens as steps on a grid:
    - `'0'`: East/right step (x + 1)
    - `'1'`: North/up step (y + 1)

    Args:
        tokens: Iterable of step tokens (typically strings `'0'` and `'1'`).
            Any other token is ignored.

    Returns:
        tuple[list[int], list[int]]: `(x_coords, y_coords)` starting at (0, 0),
        suitable for plotting a polyline of the path.
    '''
    x_coords = [0]
    y_coords = [0]

    for i, word in enumerate(tokens):
        if word == '0':  # 0 in Dyck path (right step)
            y_coords.append(y_coords[-1])
            x_coords.append(x_coords[-1] + 1)
        elif word == '1':  # 1 in Dyck path (up step)
            y_coords.append(y_coords[-1] + 1)
            x_coords.append(x_coords[-1])
        else:
            continue  # Ignore other tokens
    return x_coords, y_coords
