"""
Copyright 2020, All rights reserved.
Author : SangJae Kang
Mail : craftsangjae@gmail.com
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from .minhash import MinHash
tqdm.pandas()


class MinHashSubGraph:
    """ Minhash 알고리즘을 바탕으로, Knowledge Graph을 몇개의 subgraph로 쪼개어주는 class

    >>> minsub = MinHashSubGraph()
    >>> minsub.group(graph)

    """

    def __init__(self,
                 num_clusters=20,
                 maximum_size=100000,
                 k_core=5,
                 random_seed=1,
                 verbose=1):
        self.num_clusters = num_clusters
        self.maximum_size = maximum_size
        self.k_core = k_core
        self.random_seed = random_seed
        self.verbose = verbose
        self.__dir__ = ['group']

    def group(self, graph: np.ndarray):
        """ graph을 subgraph로 나누어 묶는 함수
        """
        graph_df = self.graph2dataframe(graph)

        cluster_df = self.assign_cluster_ids(graph_df)

        subgraphs = self.group_by_subgraph(graph_df, cluster_df)

        subgraphs = self.trim_subgraph(subgraphs)

        return subgraphs

    def graph2dataframe(self, graph: np.ndarray):
        """ graph을 dataframe의 형태로 변환
        """
        if graph.ndim != 2 or graph.shape[1] < 2:
            raise ValueError("# ndim of graph should be 2 and graph.shape[1] should be bigger than 1")

        graph_df = pd.DataFrame(graph)
        return graph_df.rename({0: 'head', 1: 'tail'}, axis=1)

    def assign_cluster_ids(self, graph_df: pd.DataFrame):
        """ graph의 head 별로 num_sigs개 cluster id를 부여
        """
        cluster_series = self._apply_minhash(graph_df)

        cluster_df = self._cluster_series2dataframe(cluster_series)
        return cluster_df

    def _apply_minhash(self, graph_df: pd.DataFrame):
        generate_minhash = MinHash(self.num_clusters, self.random_seed)
        if self.verbose:
            return graph_df.groupby('head')['tail'].progress_apply(generate_minhash)
        else:
            return graph_df.groupby('head')['tail'].apply(generate_minhash)

    def _cluster_series2dataframe(self, cluster_series):
        """ cluster series를 cluster dataframe으로 변환
        """
        clusters = [cluster_series.apply(lambda x: x[i]) for i in range(self.num_clusters)]
        clusters_df = pd.concat(clusters, axis=1)
        clusters_df.columns = [f'cluster{i}' for i in range(self.num_clusters)]
        return clusters_df

    def group_by_subgraph(self, graph_df, cluster_df):
        """ head 별 cluster id를 바탕으로 subgraph을 묶음
        """
        subgraphs = {}
        for sig_name, cluster_series in cluster_df.iteritems():
            heads = self._merge_heads(cluster_series)
            subgraphs[sig_name] = self._extract_subgraph(graph_df, heads)
        return subgraphs

    def _merge_heads(self, cluster_series):
        """ head 노드의 갯수가 maximum_size가 넘지 않는 최대 만큼 가져오기
        """
        cluster_counts = cluster_series.value_counts().sort_values(ascending=False)
        cluster_ids = cluster_counts[cluster_counts.cumsum() < self.maximum_size].index
        if len(cluster_ids) == 0:
            cluster_ids = cluster_counts.index[:1]
        return cluster_series[cluster_series.isin(cluster_ids)].index

    def _extract_subgraph(self, graph_df, heads):
        """ head을 포함하고 있는 subgraph을 가져오기
        """
        return graph_df[graph_df['head'].isin(heads)].values

    def trim_subgraph(self, subgraphs):
        """ K-core 방식을 통해 subgraph의 크기를 추림
        """
        if self.verbose:
            for name, subgraph in tqdm(subgraphs.items()):
                subgraphs[name] = self._kcore_sampling(subgraph)
        else:
            for name, subgraph in subgraphs.items():
                subgraphs[name] = self._kcore_sampling(subgraph)
        return subgraphs

    def _kcore_sampling(self, graph):
        """ k개 이하의 약한 연결을 보이는 노드들(head&tail)을 추림
        """
        if self.k_core <= 0:
            return graph
        df = self.graph2dataframe(graph)

        head_counts = df['head'].value_counts()
        tail_counts = df['tail'].value_counts()

        df = df[df['head'].isin(head_counts[head_counts >= self.k_core].index)
                & df['tail'].isin(tail_counts[tail_counts >= self.k_core].index)]

        return df.value

