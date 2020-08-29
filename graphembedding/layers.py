"""
Copyright 2020, All rights reserved.
Author : SangJae Kang
Mail : craftsangjae@gmail.com
"""
from tensorflow.keras.layers import Layer
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects


class ComplExDotScore(Layer):
    """ complEx Scoring Function
        - Based on Hermitian (or sesquilinear) dot product
        - score = Re(<relation, subject, object>)
        - Embedding의 구성
           * embed[:,:len(embed)//2] : real-value
           * embed[:,len(embed)//2:] : imaginary-value
    """

    def __init__(self, n3_reg=0., **kwargs):
        self.n3_reg = n3_reg
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        sub, rel, obj = inputs
        self.apply_regularization(inputs)

        re_sub, im_sub = tf.split(sub, 2, axis=-1)
        re_rel, im_rel = tf.split(rel, 2, axis=-1)
        re_obj, im_obj = tf.split(obj, 2, axis=-1)

        return K.sum(re_rel * re_sub * re_obj
                     + re_rel * im_sub * im_obj
                     + im_rel * re_sub * im_obj
                     - im_rel * im_sub * re_obj, axis=-1)

    def apply_regularization(self, inputs):
        if self.n3_reg:
            sub, rel, obj = inputs
            n3 = K.mean(
                K.sum(K.abs(sub) ** 3 + K.abs(rel) ** 3 + K.abs(obj) ** 3, axis=1))
            self.add_loss(self.n3_reg * n3)

    def get_config(self):
        config = super().get_config()
        config.update({
            "n3_reg": self.n3_reg
        })
        return config


class TransEScore(Layer):
    """ TransE Scoring Function
    """
    def __init__(self,
                 ord='l1',
                 margin=1,
                 **kwargs):
        assert ord in ('l1', 'l2')
        self.ord = ord
        self.margin = margin
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        pos_sub_embed = K.l2_normalize(inputs[0], axis=1)
        neg_sub_embed = K.l2_normalize(inputs[1], axis=1)
        pos_obj_embed = K.l2_normalize(inputs[2], axis=1)
        neg_obj_embed = K.l2_normalize(inputs[3], axis=1)
        rel_embed = inputs[4]

        pos_score = self._score(pos_sub_embed+rel_embed, pos_obj_embed)
        neg_score = self._score(neg_sub_embed+rel_embed, neg_obj_embed)

        loss = K.maximum(self.margin + pos_score - neg_score, 0.)
        return loss

    def _score(self, src_embed, dst_embed):
        if self.ord == 'l1':
            return K.sum(K.abs(src_embed - dst_embed), 1)
        else:
            return K.sum(K.square(src_embed - dst_embed), 1)

    def get_config(self):
        config = super().get_config()
        config.update({"ord": self.ord, "margin":self.margin})
        return config


get_custom_objects().update(
    {"ComplexDotScore": ComplExDotScore, "TransEScore": TransEScore})
