
import numpy as np
import networkx as nx



def qc_to_nxgraph(circuit):
    
    #initial nx graph
    nx_graph = nx.Graph()

    #add node to the graph
    # for i in range(circuit.num_qubits):
    #     nx_graph.add_node(i)
    qubit_key = {}
    for qubit in circuit.qubits:
        #add qubit to graph only using index for readability
        nx_graph.add_node(len(qubit_key))
        qubit_key[qubit] = len(qubit_key)

    for op in circuit.data:

        #skip if op is barrier
        if op[0].name == "barrier":
            continue

        # print(op[1])
        # if op[0].name == "cx":
        if op[0].num_qubits == 2:
            #control qubit
            c = qubit_key.get(op[1][0])
            t = qubit_key.get(op[1][1])
            if nx_graph.has_edge(c, t):
                #add 1 weight for existing edge
                new_weight = nx_graph.get_edge_data(c, t).get("weight") + 1
                nx_graph.add_edge(c, t, weight=new_weight)
            else:
                #intialize the edge
                nx_graph.add_edge(c, t, weight=1)
        elif op[0].num_qubits > 2:
            raise "More than 2-qubit gate is not supported. Please decompose your circuit."
    
    return nx_graph

def max_edges_spanning_tree(qc_graph, max_connection = 2):
    pass
    #initial new spanning tree graph without edge
    qc_spgraph = nx.Graph()

    for node in qc_graph.nodes:
        qc_spgraph.add_node(node)

    sorted_edge = sorted(qc_graph.edges(data=True), key=lambda edge: edge[2]['weight'], reverse=True)

    for edge in sorted_edge:
        n1, n2, weight = edge
        #extract weight
        weight = weight["weight"]

        #check neighbors of each node
        if len(qc_spgraph[n1]) >= max_connection or len(qc_spgraph[n2]) >= max_connection:
            continue

        #check if edge connection would create a loop
        if check_edge_connection_is_loop(qc_spgraph, n1, n2):
            continue

        # print(qc_spgraph.edges, n1, n2)
        qc_spgraph.add_edge(n1, n2, weight = weight)
    
    #check if the graph is a path
    while not nx.is_connected(qc_spgraph):
        #if graph is not a path, then connect edge of the subgraphs
        single_edge_nodes = []
        #get all node with one edge
        for node in qc_spgraph.nodes:
            if len(qc_spgraph[node]) < 2:
                single_edge_nodes.append(node)
        
        n1 = single_edge_nodes.pop()
        for n2 in single_edge_nodes:
            #check if edge connection would create a loop
            if check_edge_connection_is_loop(qc_spgraph, n1, n2):
                continue

            #there is no weight since there is no correlation
            qc_spgraph.add_edge(n1, n2, weight = 0)
            break

    return qc_spgraph

def check_edge_connection_is_loop(graph, n1, n2):
    """Checks if an edge connection would create a loop or not a spanning tree in a networkx graph.

    Args:
        G: A networkx graph.
        n1: node 1 for connection
        n2: node 2 for connection

    Returns:
        True if the edge connection would create a loop, False otherwise.
    """
    tmp_graph = graph.copy()
    tmp_graph.add_edge(n1, n2)

    #check if node are in the same subgraph
    #if both nodes are in the same subgraph, then adding edge will form a loop
    if n2 in nx.node_connected_component(graph, n1):
        return True

    if nx.is_connected(tmp_graph) and not nx.is_tree(tmp_graph):
    # if nx.is_tree(tmp_graph):
        return True
    
    return False

def check_loop_test(graph, n1, n2):
    
    tmp_graph = nx.Graph()
    # succ = graph[n1]
    # each node here can only have 1 neighbor
    n1_tmp = n1
    n1_neighbor = [n for n in graph[n1_tmp]] #get neightbor node

    #get all connected node from n1 to tmp_graph
    # def connect_neighbor(node, n_neighbor, tmp_graph):

    #     if not tmp_graph.has_edge(node, n_neighbor):
    #         tmp_graph.add_edge(node, n_neighbor)
    
    while len(n1_neighbor) > 0:
        n1_neighbor = n1_neighbor[0] #always has 1 neighbor

        if tmp_graph.has_edge(n1_tmp, n1_neighbor):
            continue
        #append edge
        tmp_graph.add_edge(n1_tmp, n1_neighbor)
        
        #update node for next neighbor append
        n1_tmp = n1_neighbor
        n1_neighbor = [n for n in graph[n1_tmp]]

    
    tmp_graph.add_edge(n1, n2)
    if nx.is_connected(tmp_graph) and not nx.is_tree(tmp_graph):
        return True
    else:
        return False

def generate_sycamore_toplogy_graph(row, col):
    
    all_qubit = np.array([i for i in range(row * col)])

    # sycamore_matrix = np.full((9,6), -1)
    # row = 9
    # col = 6
    sycamore_matrix = all_qubit.reshape(row,col)
    
    #add qubit to sycamore graph
    sycamore_graph = nx.Graph()
    for qubit in all_qubit:
        sycamore_graph.add_node(qubit)

    for nrow in range(row-1):
        for ncol in range(col):

            n1 = sycamore_matrix[nrow][ncol]

            if nrow % 2 == 0:
                if ncol < col:
                    n2_right = sycamore_matrix[nrow+1][ncol]
                    sycamore_graph.add_edge(n1, n2_right, weight=0)

                if ncol >= 1:
                    n2_left = sycamore_matrix[nrow+1][ncol-1]
                    sycamore_graph.add_edge(n1, n2_left, weight=0)
            else:
                n2_left = sycamore_matrix[nrow+1][ncol]
                sycamore_graph.add_edge(n1, n2_left, weight=0)

                if ncol < col-1:
                    n2_right = sycamore_matrix[nrow+1][ncol+1]
                    sycamore_graph.add_edge(n1, n2_right, weight=0)

    return sycamore_graph
                 

def generate_ibm_tokyo_topology_graph():
    all_qubit = np.array([i for i in range(20)])

    #ibm tokyo matrix
    row = 4
    col = 5
    tokyo_matrix = all_qubit.reshape(row, col)

    tokyo_graph = nx.Graph()
    for qubit in all_qubit:
        tokyo_graph.add_node(qubit)

    for nrow in range(row):
        for ncol in range(col):
            n1 = tokyo_matrix[nrow][ncol]
            
            if ncol < col - 1:
                n2_right = tokyo_matrix[nrow][ncol+1]
                tokyo_graph.add_edge(n1, n2_right, weight=0)
            
            if nrow < row-1:
                n2_bottom = tokyo_matrix[nrow+1][ncol]
                tokyo_graph.add_edge(n1, n2_bottom, weight=0)

    #manually add diagonal edges
    diagonal_edges = [(1,7), (2,6), (3,9), (4,8),
                      (5,11), (6,10), (7,13), (8,12),
                      (11,17), (12,16), (13,19), (14,18)]

    for n1,n2 in diagonal_edges:
        tokyo_graph.add_edge(n1, n2, weight = 0)
    
    return tokyo_graph