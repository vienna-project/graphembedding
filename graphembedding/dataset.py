"""
Copyright 2020, All rights reserved.
Author : SangJae Kang
Mail : craftsangjae@gmail.com
"""
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle


class TransEDataset:
    def __init__(self, triplets: np.ndarray):
        """
        :param:
          triplets:
             (N, 3) 크기의 np.ndarray로,
             행 별로 (head, relation, tail)로 되어 있어야 함
               - 1th column : head(subject)
               - 2nd column : relation
               - 3rd column : tail(object)
        """
        assert (triplets.ndim == 2 and triplets.shape[1] == 3), (
            "행렬은 N x 3의 크기를 가지고 있어야 합니다.")

        self.nodes = get_nodes_from_triplets(triplets)
        self.edges = get_edges_from_triplets(triplets)
        self.enc_triplets = encode_triplets(triplets, self.nodes, self.edges)

    def __call__(self, batch_size):
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        dataset = shuffle_and_create_dataset(self.enc_triplets)

        def _pipeline(edge_id):
            nonlocal dataset
            return (dataset
                    .filter(lambda x: x['relation'] == edge_id)
                    .batch(batch_size)
                    .map(self.sampler)
                    .prefetch(AUTOTUNE))

        return (tf.data.Dataset.range(len(self.edges))
                .interleave(_pipeline,
                            num_parallel_calls=AUTOTUNE,
                            deterministic=False))

    @staticmethod
    def sampler(triplets):
        """Edge Negative Sampling strategy in transE Model
        """
        t = triplets
        p_sub, p_obj, rel = t['subject'], t['object'], t['relation']
        n_sub, n_obj = corrupt_head_or_tail(p_sub, p_obj)
        return {"pos_subject": p_sub, "neg_subject": n_sub,
                "pos_object": p_obj, "neg_object": n_obj,
                "relation": rel}


class ComplExDataset:
    def __init__(self, triplets: np.ndarray):
        """
        :param:
          triplets:
             (N, 3) 크기의 np.ndarray로,
             행 별로 (head, relation, tail)로 되어 있어야 함
               - 1th column : head(subject)
               - 2nd column : relation
               - 3rd column : tail(object)
        """
        assert (triplets.ndim == 2 and triplets.shape[1] == 3), (
            "행렬은 N x 3의 크기를 가지고 있어야 합니다.")

        self.nodes = get_nodes_from_triplets(triplets)
        self.edges = get_edges_from_triplets(triplets)
        self.enc_triplets = encode_triplets(triplets, self.nodes, self.edges)

    def __call__(self, batch_size, num_negs):
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        dataset = shuffle_and_create_dataset(self.enc_triplets)

        def _pipeline(edge_id):
            nonlocal dataset
            return (dataset
                    .filter(lambda x: x['relation'] == edge_id)
                    .batch(batch_size)
                    .map(self.sampler(num_negs))
                    .prefetch(AUTOTUNE))

        return (tf.data.Dataset.range(len(self.edges))
                .interleave(_pipeline,
                            num_parallel_calls=AUTOTUNE,
                            deterministic=False))

    @staticmethod
    def sampler(num_negs):
        """Edge Negative Sampling strategy in complEx Model
        params :
            * num_neg: 1 positive sample 당 negative 비율
        """

        def func(triplet):
            t = triplet
            p_rel, n_rel = t['relation'], tf.tile(t['relation'], [num_negs])
            p_sub, n_sub = t['subject'], tf.tile(t['subject'], [num_negs])
            p_obj, n_obj = t['object'], tf.tile(t['object'], [num_negs])

            n_sub, n_obj = corrupt_head_or_tail(n_sub, n_obj)

            inputs = {'relation': tf.concat([p_rel, n_rel], axis=-1),
                      'subject': tf.concat([p_sub, n_sub], axis=-1),
                      'object': tf.concat([p_obj, n_obj], axis=-1)}

            p_labels, n_labels = tf.ones_like(p_rel), tf.zeros_like(n_rel)
            labels = tf.concat([p_labels, n_labels], axis=-1)

            return inputs, labels
        return func


class ImplExDataset:
    def __init__(self, triplets: np.ndarray):
        """
        :param:
          triplets:
             (N, 4) 크기의 np.ndarray로,
             행 별로 (head, relation, tail, relation_strength)로 되어 있어야 함
               - 1th column : head(subject)
               - 2nd column : relation
               - 3rd column : tail(object)
               - 4th column : relation_strength
        """
        assert (triplets.ndim == 2 and triplets.shape[1] == 4), (
            "행렬은 N x 4의 크기를 가지고 있어야 합니다.")

        self.nodes = get_nodes_from_triplets(triplets)
        self.edges = get_edges_from_triplets(triplets)
        self.counts = triplets[:, -1].astype(np.float32)
        self.enc_triplets = (*encode_triplets(triplets, self.nodes, self.edges), self.counts)

    def __call__(self, batch_size, num_negs):
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        dataset = shuffle_and_create_dataset(self.enc_triplets)

        def _pipeline(edge_id):
            nonlocal dataset
            return (dataset
                    .filter(lambda x: x['relation'] == edge_id)
                    .batch(batch_size)
                    .map(self.sampler(num_negs))
                    .prefetch(AUTOTUNE))

        return (tf.data.Dataset.range(len(self.edges))
                .interleave(_pipeline,
                            num_parallel_calls=AUTOTUNE,
                            deterministic=False))

    @staticmethod
    def sampler(num_negs):
        """Edge Negative Sampling strategy in complEx Model
        params :
            * num_neg: 1 positive sample 당 negative 비율
        """

        def func(triplet):
            t = triplet
            p_rel, n_rel = t['relation'], tf.tile(t['relation'], [num_negs])
            p_sub, n_sub = t['subject'], tf.tile(t['subject'], [num_negs])
            p_obj, n_obj = t['object'], tf.tile(t['object'], [num_negs])

            n_sub, n_obj = corrupt_head_or_tail(n_sub, n_obj)

            p_cnt, n_cnt = t['count'], tf.tile(tf.zeros_like(t['count']), [num_negs])

            p_labels, n_labels = tf.ones_like(p_rel), tf.zeros_like(n_rel)
            labels = tf.concat([p_labels, n_labels], axis=-1)

            inputs = {'relation': tf.concat([p_rel, n_rel], axis=-1),
                      'subject': tf.concat([p_sub, n_sub], axis=-1),
                      'object': tf.concat([p_obj, n_obj], axis=-1),
                      'count': tf.concat([p_cnt, n_cnt], axis=-1),
                      'label': labels}

            return inputs, labels
        return func


def get_nodes_from_triplets(triplets):
    return list(set(triplets[:, 0]) | set(triplets[:, 2]))


def get_edges_from_triplets(triplets):
    return list(set(triplets[:, 1]))


def encode_triplets(triplets, nodes, edges):
    node2id = {node: i for i, node in enumerate(nodes)}
    edge2id = {edge: i for i, edge in enumerate(edges)}

    subs = np.vectorize(node2id.__getitem__)(triplets[:, 0])
    rels = np.vectorize(edge2id.__getitem__)(triplets[:, 1])
    objs = np.vectorize(node2id.__getitem__)(triplets[:, 2])
    return subs, rels, objs


def shuffle_and_create_dataset(encoded_triplets):
    if len(encoded_triplets) == 3:
        subs, rels, objs = shuffle(*encoded_triplets)
        tensors = {"subject": subs, "object": objs, "relation": rels}
        return tf.data.Dataset.from_tensor_slices(tensors)
    elif len(encoded_triplets) == 4:
        subs, rels, objs, counts = shuffle(*encoded_triplets)
        tensors = {"subject": subs, "object": objs, "relation": rels, "count": counts}
        return tf.data.Dataset.from_tensor_slices(tensors)


def corrupt_head_or_tail(heads, tails):
    """ 50% 확률로 head 혹은 tail을 corrupt
    """
    h_flag = tf.random.uniform(tf.shape(heads)) < 0.5

    neg_heads = tf.where(
        h_flag, heads, tf.random.shuffle(heads))
    neg_tails = tf.where(
        h_flag, tf.random.shuffle(tails), tails)
    return neg_heads, neg_tails
