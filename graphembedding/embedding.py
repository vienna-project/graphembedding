"""
Copyright 2020, All rights reserved.
Author : SangJae Kang
Mail : craftsangjae@gmail.com
"""
import pandas as pd
import numpy as np
from .dataset import TransEDataset, ComplExDataset
from .models import create_transEModel, create_complExModel


"""
How to use? 

It's simple. 
example code is below.

>>> github_dataset = load_github()
>>> triplets = github_dataset[['subject','relation','object']].values #
>>> node_embedding, edge_embedding = complEx(triplets)

That's all.
"""


def transE(triplets:np.ndarray,
           embed_size=50,
           ord='l1',
           margin=1,
           learning_rate=2e-1,
           batch_size=1024,
           num_epochs=50,
           callbacks=None,
           keras_model=None,
           return_keras_model=False,
           verbose=1):
    """
    Node & Edge Embedding using transE Algorithm.

    reference : Translating Embeddings for Modeling Multi-relational Data(2013)

    :param triplets: 각 행이 (subject, relation, object)으로 이루어진 (N,3) np.ndarray
    :param embed_size: 임베딩 벡터의 크기
    :param ord: 손실 함수로 'l1' 혹은 'l2' 중 택 1.
    :param margin: 손실 함수 내 margin의 크기
    :param learning_rate: 학습률
    :param batch_size: 배치 크기
    :param num_epochs: 학습 횟수
    :param callbacks: tf.keras.callbacks
    :param keras_model: 만약 tf.keras.Model의 인스턴스를 지정한다면, 해당 모형으로 초기화한후 학습
    :param return_keras_model: tf.keras.Model을 반환할지 유무
    :param verbose: 0 or 1. Verbosity mode. 0 = silent, 1 = progress bar.

    :return:
        - node_embedding : node(subject&object)에 대한 임베딩 행렬
        - edge_embedding : edge(relation)에 대한 임베딩 행렬
        - [model] : 학습에 이용된 tf.keras.Model
    """
    dataset = TransEDataset(triplets)

    if keras_model is None:
        model = create_transEModel(len(dataset.nodes), len(dataset.edges),
                                   embed_size, ord, margin, learning_rate)
    else:
        model = keras_model

    try:
        for i in range(num_epochs):
            model.fit(dataset(batch_size), verbose=verbose,
                      epochs=i+1, initial_epoch=i,
                      shuffle=False, callbacks=callbacks)
    except KeyboardInterrupt:
        pass

    node_embedding, edge_embedding = weight2embedding(model, dataset)

    if return_keras_model:
        return node_embedding, edge_embedding, model
    else:
        return node_embedding, edge_embedding


def complEx(triplets:np.ndarray,
            embed_size=50,
            n3_reg=1e-3,
            learning_rate=5e-1,
            num_negs=20,
            batch_size=1024,
            num_epochs=50,
            callbacks=None,
            keras_model=None,
            return_keras_model=False,
            verbose=1):
    """
    Node & Edge Embedding using complEx Algorithm.

    reference : Complex Embeddings for Simple Link Prediction(2016)

    :param triplets: 각 행이 (subject, relation, object)으로 이루어진 (N,3) np.ndarray
    :param embed_size: 임베딩 벡터의 크기
    :param n3_reg: tensor nuclear 3-norms 정규화의 크기
            - reference : Canonical Tensor Decomposition for Knowledge Base Completion(2018)
    :param learning_rate:  손실 함수 내 margin의 크기
    :param num_negs: Negative Sampling의 갯수
    :param batch_size: 배치 크기
    :param num_epochs: 학습 횟수
    :param callbacks: tf.keras.callbacks
    :param keras_model: 만약 tf.keras.Model의 인스턴스를 지정한다면, 해당 모형으로 초기화한후 학습
    :param return_keras_model: tf.keras.Model을 반환할지 유무
    :param verbose: 0 or 1. Verbosity mode. 0 = silent, 1 = progress bar.

    :return:
        - node_embedding : node(subject&object)에 대한 임베딩 행렬
        - edge_embedding : edge(relation)에 대한 임베딩 행렬
        - [model] : 학습에 이용된 tf.keras.Model
    """
    dataset = ComplExDataset(triplets)

    if keras_model is None:
        model = create_complExModel(len(dataset.nodes), len(dataset.edges),
                                    embed_size, n3_reg, learning_rate)
    else:
        model = keras_model

    try:
        for i in range(num_epochs):
            model.fit(dataset(batch_size, num_negs),
                      epochs=i+1, initial_epoch=i, verbose=verbose,
                      class_weight={1: 1., 0: 1 / num_negs},
                      shuffle=False, callbacks=callbacks)
    except KeyboardInterrupt:
        pass

    node_embedding, edge_embedding = weight2embedding(model, dataset)

    if return_keras_model:
        return node_embedding, edge_embedding, model
    else:
        return node_embedding, edge_embedding


def weight2embedding(model, dataset):
    node_embedding = pd.DataFrame(model.get_layer("node_embedding").get_weights()[0])
    node_embedding.index = dataset.nodes

    edge_embedding = pd.DataFrame(model.get_layer("edge_embedding").get_weights()[0])
    edge_embedding.index = dataset.edges
    return node_embedding, edge_embedding

