"""
Copyright 2020, All rights reserved.
Author : SangJae Kang
Mail : craftsangjae@gmail.com
"""
import numpy as np
import struct
import hashlib


UINT64_MAX = 2 ** 64 - 1


class MinHash:
    """ 집합에 대한 MinHash Value을 생성하는 class
    """

    def __init__(self, num_sigs, seed=1):
        self.seed = seed
        self.num_sigs = num_sigs
        self.perms = self.permutation()

    def __call__(self, keys):
        """ 해당 집합(keys)에 대한 minhash 값을 생성"""
        return (np.stack([self.signature(key) for key in keys])).min(axis=0)

    def permutation(self):
        """ minhash를 계산하기 위한 랜덤 수열 생성 """
        generator = np.random.RandomState(self.seed)
        A = np.array([generator.randint(0, UINT64_MAX, dtype=np.uint64)
                      for _ in range(self.num_sigs)])
        B = np.array([generator.randint(1, UINT64_MAX, dtype=np.uint64)
                      for _ in range(self.num_sigs)])
        return A, B

    def signature(self, x: int):
        """ 해당 key(X)에 대한 signature값을 생성"""
        hash_value = struct.unpack(
            '<I', hashlib.sha1(str(x).encode()).digest()[:4])[0]
        A, B = self.perms
        return np.array((A * hash_value + B) % UINT64_MAX, dtype=np.uint64)

