"""
Copyright 2020, All rights reserved.
Author : SangJae Kang
Mail : craftsangjae@gmail.com
"""
import numpy as np
from tqdm import tqdm


def kcore_sampling(graph:np.ndarray, k_core=5):
    if k_core <= 0:
        return graph

    with tqdm(desc="k-core sampling") as pbar:
        while True:
            prev_counts = len(graph)
            if prev_counts == 0:
                raise ValueError("No data remains")

            heads, tails = graph[:, 0], graph[:, -1]

            heads_over_k = [
                k for k, c in zip(*np.unique(heads, return_counts=True))
                if c >= k_core]
            tails_over_k = [
                k for k, c in zip(*np.unique(tails, return_counts=True))
                if c >= k_core]
            mask = np.in1d(heads, heads_over_k) & np.in1d(tails, tails_over_k)
            graph = graph[mask]

            if prev_counts == len(graph):
                # 변화가 없으면 종료
                return graph
            pbar.update(1)
