from scipy.sparse import coo_matrix


def group_to_sparse_vector(grp, max_id, binary=False):
    key = grp[0]
    if binary:
        data = [1 for k in grp[1]]
    else:
        data = [k[1] for k in grp[1]]
    cols = [k[0] for k in grp[1]]
    rows = [0] * len(grp[1])
    vector = coo_matrix((data, (rows, cols)), shape=(1, max_id))
    return key, vector


def create_coocc_pairs(fragment):
    return [(fragment[0], k) for k in fragment[1:]] + [(k, fragment[0]) for k in fragment[1:]]
