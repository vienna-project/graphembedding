## GraphEmbedding

Python Graph Embedding Libary for Knowledge graph

This project provides Tensorflow2.0 implementatinons of several different popular graph embeddings for knowledge graph.

* [transE](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf)
* [complEx](http://proceedings.mlr.press/v48/trouillon16.pdf)

#### Installation: 

`graphembedding` will be released on pypi soon.

````bash
python setup.py install
````


#### Basic Usages:

It's simple. example code is below.
The embedding object is returned as `pd.Dataframe`, so it can be used easily.


````python
from graphembedding.playground import load_github
from graphembedding import complEx, transE

# Load Sample dataset 
github_dataset = load_github() 
triplets = github_dataset[['subject','relation','object']].values

# That's all. One line code.
node_embedding, edge_embedding = complEx(triplets) 

# if you wanna use transE,
# node_embedding, edge_embedding = transE(triplets) 

````
