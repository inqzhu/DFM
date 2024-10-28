# Distributional Factor Model (DFM)


DFM is developped for clustering objects with repeated observations. With the rapid development of information technology, clustering analyses are now facing with challenges in increasing scales of datasets and complex data scenarios. In many
applications, there exist inherent relations or hierarchical nesting structures among observations. In order to make full use of observed data, we consider distribution function as a useful tool to describe clustering objects avoiding information loss. Then, we propose distributional factor model (DFM) based on Gaussian mixture model. With the help of DFM, we decompose the observed data to two parts. The first part is a set of common factors, which are expressed as Gaussian components. These reflect the common patterns among the whole dataset. The second part is a loading matrix, each row of which is a loading vector corresponding to an individual object. This describes an object’s heterogeneous features. Then clustering could be conducted based on the loading vectors for all objects. 

More details of DFM can be found at: https://doi.org/10.19343/j.cnki.11-1302/c.2024.06.012
 
https://link.cnki.net/doi/10.19343/j.cnki.11-1302/c.2024.06.012

https://link.oversea.cnki.net/doi/10.19343/j.cnki.11-1302/c.2024.06.012

To use DFM, please import the code `dfm.py`. The file `example.py` provides an example.
```Python
    import dfm

    # 0.generate simulated dataset
    ...

    # 1.load simulated dataset
    ...

    # 2.DFM modeling

    # specify number of clusters
    K = 3
    # specify number of Gaussian components
    G = 2
    key = 'ID' # specify the column of the key
    value = ['x1', 'x2', 'x3'] # specify the columns of the values
    worker = dfm.DFM(K, G, key, value) # initialize DFM
    worker.load(data) # loading data
    worker.init_state() # obtain initial state
    T_iter = 10 # specify the threshold of iterations

    # start training
    for t in range(T_iter):
        # e-step
        worker.e_step() 
        # m-step
        worker.m_step()

    # 3.evaluate the clustering performance
    # obtain loading vectors for all objects
    X = [] 
    for i in range(len(data)):
        line = data.iloc[i]
        X.append( worker.phis[ line[key] ] )
    X = np.array(X)
    
    # K-means clustering for vectors of objects
    clusterer = KMeans(n_clusters=K, random_state=0).fit(X)
    cluster_labels=clusterer.labels_

    # performance evaluation 
    ...
```

The structure of the input CSV should be, for example:
| Merchant_id | x1 | x2 |
| ----------- | ----------- | 
| mer001 | 100 | 100 |
| mer001 | 105 | 80 |
| mer002 | 50 | 40 |
| mer002 | 45 |50 |
| ... | ... | ... |


Here `key` and `values` are `Merchant_id` and `['x1','x2']`, respectively.

The loading vectors outputted by DFM (the attribute `phis`, e.g., `worker.phis`) is organized as a dictionary with a key-value structure:

```
{ 
'mer001': [0.2, 0.1, ...],
'mer002': [0.1, 0.9, ...],
...
}
```

The GMM components outputeed by DFM are `gmm`, e.g., `worker.gmm`.
