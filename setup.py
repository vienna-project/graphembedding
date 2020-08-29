"""
Copyright 2020, All rights reserved.
Author : SangJae Kang
Mail : craftsangjae@gmail.com
"""
import io
from setuptools import find_packages, setup


def long_description():
    with io.open("README.md", 'r', encoding='utf-8') as f:
        readme = f.read()
    return readme


setup(name='graphembedding',
      version='0.1',
      description='Python Graph Embedding Library for Knowledge graph',
      long_description=long_description(),
      url="https://github.com/vienna-project/graphembedding",
      author='craftsangjae',
      author_email='craftsangjae@gmail.com',
      license='MIT',
      packages=find_packages(exclude=['examples']),
      python_requires=">=3",
      install_requires=["pandas", "tqdm",
                        "numpy", "scikit_learn",
                        "tensorflow>=2.2.0"],
      classifiers=['Programming Language :: Python :: 3'],
      zip_safe=False)
