def test_membership():
    from ai4bmr_learn.clustering.cluster import get_membership_from_data
    import numpy as np
    from itertools import product

    data = np.random.rand(1_000, 20)

    # methods = ['scanpy', 'rapids-sc']
    #
    # for method in methods:
    #     membership = get_membership_from_data(data, method=method)

    graph_engines = ['cuml', 'rapids-sc', 'scanpy', 'sklearn']
    graph_engines = ['cuml', 'scanpy', 'sklearn']
    membership_engines = ['igraph', 'leiden']
    for graph_engine, membership_engine in product(graph_engines, membership_engines):
        membership = get_membership_from_data(data, graph_engine=graph_engine, membership_engine=membership_engine)

test_membership()