import collections
import gc
import numpy as np
import tqdm


def bin_count(data):
    bc = np.bincount(data)
    bc = 1. / (bc + 1)  # Smoothing - if bincount is zero - very problematic!
    bc = bc / np.sum(bc)
    return bc


def get_bin_counts(dataloader, keys, n_iters=1000):
    all_values = collections.defaultdict(list)
    bincounts = {}

    # sample data
    iters = 0
    while True:
        for ix, batch in enumerate(dataloader):
            if iters >= n_iters:
                break
            for key in keys:
                values = batch[key].data.cpu().numpy()
                values = values[np.where(values != -100)].flatten().tolist()

                all_values[key].extend(values)

            iters += 1
        if iters >= n_iters:
            break

    # calculate bin counts for each target
    for key in tqdm.tqdm(keys):
        values = all_values[key]
        values = np.array(values, dtype=np.int)
        bincounts[key] = bin_count(values)

    del all_values
    gc.collect()

    return bincounts