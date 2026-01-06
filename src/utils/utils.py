def word_to_path(tokens):
    """Convert Dyckword sequence to path coordinates for plotting. North step is represented by a 1 and an East step is represented by a 0."""
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
