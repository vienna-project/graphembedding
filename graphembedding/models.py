"""
Copyright 2020, All rights reserved.
Author : SangJae Kang
Mail : craftsangjae@gmail.com
"""
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Embedding
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.initializers import RandomUniform, GlorotUniform
from tensorflow.keras.optimizers import Adagrad
from .layers import TransEScore, ComplExDotScore


def create_transEModel(num_nodes,
                       num_edges,
                       embed_size=50,
                       ord='l1',
                       margin=1,
                       learning_rate=2e-1):
    # build transE Model
    pos_sub_inputs = Input(shape=(), name='pos_subject')
    neg_sub_inputs = Input(shape=(), name='neg_subject')
    pos_obj_inputs = Input(shape=(), name='pos_object')
    neg_obj_inputs = Input(shape=(), name='neg_object')
    rel_inputs = Input(shape=(), name='relation')

    inputs = {
        "pos_subject": pos_sub_inputs,
        "neg_subject": neg_sub_inputs,
        "pos_object": pos_obj_inputs,
        "neg_object": neg_obj_inputs,
        "relation": rel_inputs
    }

    # 초기화 방식은 논문에 나와있는 방식으로 구성
    init_range = 6/np.sqrt(embed_size)
    init_op = RandomUniform(-init_range, init_range)

    node_layer = Embedding(input_dim=num_nodes,
                           output_dim=embed_size,
                           embeddings_initializer=init_op,
                           name='node_embedding')
    edge_layer = Embedding(input_dim=num_edges,
                           output_dim=embed_size,
                           embeddings_initializer=init_op,
                           name='edge_embedding')

    pos_sub = node_layer(pos_sub_inputs)
    neg_sub = node_layer(neg_sub_inputs)
    pos_obj = node_layer(pos_obj_inputs)
    neg_obj = node_layer(neg_obj_inputs)
    rel = edge_layer(rel_inputs)

    score = TransEScore(ord, margin)([pos_sub, neg_sub, pos_obj, neg_obj, rel])
    model = Model(inputs, score)

    # Compile transE Model
    model.add_loss(score)
    model.compile(optimizer=Adagrad(learning_rate))

    return model


def create_complExModel(num_nodes,
                        num_edges,
                        embed_size=50,
                        n3_reg=1e-3,
                        learning_rate=5e-1):
    # Build complEx Model
    sub_inputs = Input(shape=(), name='subject')
    obj_inputs = Input(shape=(), name='object')
    rel_inputs = Input(shape=(), name='relation')
    inputs = {"subject": sub_inputs, "object": obj_inputs, "relation": rel_inputs}

    node_layer = Embedding(input_dim=num_nodes,
                           output_dim=embed_size,
                           embeddings_initializer=GlorotUniform(),
                           name='node_embedding')
    edge_layer = Embedding(input_dim=num_edges,
                           output_dim=embed_size,
                           embeddings_initializer=GlorotUniform(),
                           name='edge_embedding')

    sub_embed = node_layer(sub_inputs)
    rel_embed = edge_layer(rel_inputs)
    obj_embed = node_layer(obj_inputs)

    outputs = ComplExDotScore(n3_reg)([sub_embed, rel_embed, obj_embed])
    model = Model(inputs, outputs, name='complEx')

    # Compile complEx Model
    loss = BinaryCrossentropy(from_logits=True, reduction='sum')
    model.compile(optimizer=Adagrad(learning_rate), loss=loss, metrics=[loss])

    return model