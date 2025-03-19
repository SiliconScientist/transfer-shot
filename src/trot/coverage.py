import numpy as np


def within_interval(y: float, lower: float, upper) -> bool:
    return lower <= y <= upper


def bin_to_indices(bin_indices, bin_edges):
    binned_samples = {i: [] for i in range(1, len(bin_edges))}
    for idx, bin_index in enumerate(bin_indices):
        if bin_index in binned_samples:  # Ensure the index is within range
            binned_samples[bin_index].append(idx)
    return binned_samples


def get_binned_indices(data, bins):
    _, bin_edges = np.histogram(data, bins=bins)
    bin_indices = np.digitize(data, bin_edges)
    binned_indices = bin_to_indices(bin_indices, bin_edges)
    return binned_indices


def get_bin_coverage(bin, y, lower, upper):
    num_in_bin = len(bin)
    num_in_bounds = 0
    for i in bin:
        if within_interval(y=y[i], lower=lower[i], upper=upper[i]):
            num_in_bounds += 1
    if num_in_bin == 0:
        pass
    else:
        coverage = num_in_bounds / num_in_bin
        return coverage


def get_fsc_metric(binned_indices, y, lower, upper):
    """
    Feature Stratified Coverage Metric
    """
    coverages = []
    for bin in binned_indices.values():
        coverage = get_bin_coverage(bin, y=y, lower=lower, upper=upper)
        coverages.append(coverage)
    return np.min(coverages)
