def remove_backclicks(path):
    """
    This only applies to paths in the WikiSpeedia dataset.
    :param path:
    :return:
    """
    tmp_path = reversed(path)
    result_path = []
    skip = 0
    for elem in tmp_path:
        if elem == '<':
            skip += 1
            continue
        if skip > 0:
            skip -= 1
            continue
        result_path.append(elem)
    return list(reversed(result_path))
