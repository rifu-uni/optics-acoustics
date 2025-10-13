import cv2

def avg_pixel_values_multiple_points(path_list, points, length, verbose=False):
    """
    For each image in path_list, compute the average BGR value over a square
    region of size `length` around each coordinate in `points`.
    Returns a dictionary: {point: list of average BGR triplets across images}.
    """
    results = {tuple(pt): [] for pt in points}

    for idx, file in enumerate(path_list):
        img = cv2.imread(str(file))
        if img is None:
            continue  # skip unreadable files

        for pt in points:
            x, y = pt
            region = img[x:x+length, y:y+length]
            avg_vals = region.mean(axis=(0, 1))
            results[tuple(pt)].append(avg_vals)

        if verbose:
            print(f"Processed image {idx+1}/{len(path_list)}: {file}")

    return results
