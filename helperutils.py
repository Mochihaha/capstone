"""
Utility functions to support customised / repeatable actions.

"""

import time
import numpy as np
from datetime import datetime
from typing import Union, Tuple, Iterable
from collections import defaultdict
from scipy.stats import spearmanr
from scipy.cluster import hierarchy


def plog(msg: str, typ: str='INFO') -> None:
    """Prints messages with timestamp, customisable message type info. prefixed when printing.
    Printed time is in UTC, non-local time."""
    
    tm = time.time()
    tm_str = datetime.utcfromtimestamp(tm).strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{typ}] UTC {tm_str} {msg}')


def get_similarity_repr(
    similarity_matrix: np.ndarray,
    cophenetic_dist: float=2.,
    grouped_idx: bool=False) -> Tuple[Union[int, float]] and list:
    """
    Group features according to a N x N similarity matrix and return only 1 from each group.
    
    @similarity_matrix: np.ndarray
        N x N matrix of features' similarity scores with each other.
        Example of such include a covariance matrix.
        
    @cophenetic_dist: float
        Distance to control number of generated feature-groups valid for X > 0.
        Lower value will result in more feature-groups, consequently more representative features
        
        Scores exceeding data range will be re-adjusted as follows:
            score > (1 + (1/3))*max                 --> score = 0.9*max
            score < (2/3)*min                       --> score = 1.1*min
            all not values within buffer conditions --> score = average(min, max)
            
    @grouped_idx: bool
        Indicate whether to return all index in their groups (list of list)
        or the first-indexed feature-index for each group (list of idx(int))
        
    ---
    Returns list of grouped features by index in matrix if `grouped_idx`==True; else list of representative features (first element in group) by their indices in matrix
    
    """
    
    corr_linkage = hierarchy.ward(similarity_matrix)
    min_dist = corr_linkage[:, 2].min()
    max_dist = corr_linkage[:, 2].max()
        # See Scipy document on structure of linkage matrix
        # @https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage
    
    # Resolve if cophenetic distance is outside of data range
    if not (min_dist <= cophenetic_dist <= max_dist):
        plog('Specified "cophenetic_dist" is outside of dendrogram range', typ='WARN')
        
        if cophenetic_dist > (1+(1/3))*max_dist:
            cophenetic_dist = .9*max_dist
            plog(f'Specifed "cophenetic_dist" exceeds (1+(1/3))*max value')
            plog(f'Reassigning to 0.9*max value at {cophenetic_dist}')
        elif cophenetic_dist < (2/3)*min_dist:
            cophenetic_dist = 1.1*min_dist
            plog(f'Specified "cophenetic_dist" undershoots (2/3)*min value')
            plog(f'Reassigning to 1.1*min value at {cophenetic_dist}')
        else:
            cophenetic_dist = 0.5 * (min_dist + max_dist)
            plog(f'Average of min/max cophenetic distance range assigned')
    
    cluster_ids = hierarchy.fcluster(corr_linkage, cophenetic_dist, criterion='distance')
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    
    # Return either all feature index in groups or first-indexed of each
    if grouped_idx:
        selected_features = [v for v in cluster_id_to_feature_ids.values()]
    else:
        selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
    
    return selected_features

