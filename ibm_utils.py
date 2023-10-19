
import numpy as np
import networkx as nx

from qiskit.circuit.random import random_circuit
# Import from Qiskit Aer noise module
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
    pauli_error, depolarizing_error, thermal_relaxation_error)

from qiskit import transpile, execute
from qiskit.transpiler import CouplingMap

def generate_random_circuit(n_qubits, n_depth, measure = True, save_file_dir = None):
    
    qc = random_circuit(n_qubits, n_depth, measure=True, max_operands=2)

    transpiled_qc = transpile(qc, basis_gates=['cx', 'id', 'rz', 'x', 'sx'], optimization_level=0)

    if save_file_dir:
        #save qc to qasm file
        transpiled_qc.qasm(filename=save_file_dir)
    
    return transpiled_qc
    
def generate_custom_coupling_map(cmap_graph):

    cmap = []
    sorted_edge = sorted(cmap_graph.edges(data=True), key=lambda edge: edge[2]['weight'], reverse=True)
    for edge in sorted_edge:
        cmap.append((edge[0], edge[1]))

    cmap = CouplingMap(cmap)
    cmap.make_symmetric()

    return cmap

    
"""
pos_matrix: grid matrix with qubit position on the matrix
cmap: Qiskit CouplingMap
qc_graph: QuantumCircuit graph weight correlation
"""
def connect_adjacent_and_diagonal_edge(pos_matrix, cmap_graph, qc_graph, always_allowed_adj = True):
    
    row, col = pos_matrix.shape

    #add edge for adjacency node
    #iterate through all row except the last row
    for nrow in range(row - 1):
        for ncol in range(col):
            n1 = pos_matrix[nrow][ncol]
            n2 = pos_matrix[nrow+1][ncol]

            if n1 == -1 or n2 == -1:
                continue

            #Connection will not be added if no edge if found
            if qc_graph.has_edge(n1, n2) or always_allowed_adj:
                # continue
                # weight = qc_graph.get_edge_data(n1, n2)["weight"]
                weight = qc_graph.get_edge_data(n1, n2)

                if weight:
                    weight = weight["weight"] #extract weight
                else:
                    weight = 0

                cmap_graph.add_edge(n1, n2, weight = weight)

    #add edge for diagonal node
    #iterate through all row except the last row
    for nrow in range(row - 1):
        for ncol in range(col):
            #get the node of the current position
            n1 = pos_matrix[nrow][ncol]

            # if ncol - 1 is < 0, there is no left diagonal
            # if ncol - 1 >= 0 or ncol + 1 > col:
            if ncol - 1 >= 0:
                n2_left_diagonal = pos_matrix[nrow+1][ncol-1]
                if qc_graph.has_edge(n1, n2_left_diagonal):
                    weight = qc_graph.get_edge_data(n1, n2_left_diagonal)["weight"]
                    cmap_graph.add_edge(n1, n2_left_diagonal, weight = weight)

            # if ncol + 1 > col, there is no right diagonal
            if ncol + 1 < col:
                n2_right_diagonal = pos_matrix[nrow+1][ncol+1]
                if qc_graph.has_edge(n1, n2_right_diagonal):
                    weight = qc_graph.get_edge_data(n1, n2_right_diagonal)["weight"]
                    cmap_graph.add_edge(n1, n2_right_diagonal, weight = weight)
    return cmap_graph

def eliminate_diagonal_edges_with_constraints(pos_matrix, cmap_graph):

    row, col = pos_matrix.shape
    g1_edges = nx.Graph() #init empty graph for group 1 edges
    g2_edges = nx.Graph() #init empty graph for group 2 edges

    for nrow in range(row -1):

        #the g1 and g2 will be flipped in each row 
        if nrow > 0:
            g1_edges, g2_edges = g2_edges, g1_edges

        for ncol in range(col - 1):
            #pair 1 of the diagonal edge
            p1_left = pos_matrix[nrow][ncol]
            p1_right = pos_matrix[nrow+1][ncol+1]

            #pair 2 of the diagonal edge
            p2_right = pos_matrix[nrow][ncol+1]
            p2_left = pos_matrix[nrow+1][ncol]
            
            #weight accumulation for the first group
            if ncol % 2 == 0:
                if cmap_graph.has_edge(p1_left, p1_right):
                    weight = cmap_graph.get_edge_data(p1_left, p1_right)["weight"]
                    g1_edges.add_edge(p1_left, p1_right, weight = weight)

                if cmap_graph.has_edge(p2_left, p2_right):
                    weight = cmap_graph.get_edge_data(p2_left, p2_right)["weight"]
                    g1_edges.add_edge(p2_left, p2_right, weight = weight)

            #weight accumulation for the second group
            else:
                if cmap_graph.has_edge(p1_left, p1_right):
                    weight = cmap_graph.get_edge_data(p1_left, p1_right)["weight"]
                    g2_edges.add_edge(p1_left, p1_right, weight = weight)

                if cmap_graph.has_edge(p2_left, p2_right):
                    weight = cmap_graph.get_edge_data(p2_left, p2_right)["weight"]
                    g2_edges.add_edge(p2_left, p2_right, weight = weight)

    #accumulate the weight and remove diagonal edges that has less weight
    #if g1_edges has more weight, remove all diagonal edges in g2_edges
    if g1_edges.size(weight="weight") > g2_edges.size(weight="weight"):
        for edge in g2_edges.edges:
            n1, n2 = edge
            cmap_graph.remove_edge(n1, n2)
    #if g1_edges has less weight, remove all diagonal edges in g1_edges
    else:
        for edge in g1_edges.edges:
            n1, n2 = edge
            cmap_graph.remove_edge(n1, n2)

    return cmap_graph

def generate_grid_matrix(row, col, qc_mpgraph):
    
    grid_matrix = np.full((row, col), -1)
    start_path = None
    #get max graph path first or last node
    for node in qc_mpgraph.nodes:
        if len(qc_mpgraph[node]) == 1:
            start_path = node
            break
    
    if start_path is None:
        raise ValueError("Invalid max path path: qc_mpgraph")

    grid_matrix[0][0] = start_path

    for nrow in range(row):
        #start row in normal order
        if nrow % 2 == 0:
            for ncol in range(col):
                if grid_matrix[nrow][ncol] != -1:
                    previous_node = grid_matrix[nrow][ncol]
                    continue
                
                neighbor = get_neighbor(previous_node, qc_mpgraph, grid_matrix)

                # TODO: Solve the constraint when there is no more qubit to append
                # to grid graph
                if len(neighbor) == 0:
                    break

                grid_matrix[nrow][ncol] = neighbor[0]
                previous_node = neighbor[0]

        #start row in reverse order
        else:
            for ncol in reversed(range(col)):
                if grid_matrix[nrow][ncol] != -1:
                    previous_node = grid_matrix[nrow][ncol]
                    continue

                neighbor = get_neighbor(previous_node, qc_mpgraph, grid_matrix)
                if len(neighbor) == 0:
                    break
                
                grid_matrix[nrow][ncol] = neighbor[0]
                previous_node = neighbor[0]
    
    return grid_matrix
"""
node: node to get neighbor
graph: graph to find the neighbor of the node
exclude_list: neighor that are in exclude list will be remove from the return neighbor
"""
def get_neighbor(node, graph, exclude_list = None):
    
    exclude_list = exclude_list.flatten()

    neighbors = []

    for node in graph[node]:
        
        #skip node in exclude list
        if node in exclude_list:
            continue
        
        neighbors.append(node)

    return neighbors


def ibm_executor(circuit, cmap, optimization_level, backend = AerSimulator(), noise_model="", noise_level=0, init_layout = None):

    #setup
    # backend = Aer.get_backend("qasm_simulator")
    # backend = AerSimulator()
    if noise_model.lower() == "depolarizing":
        # noise_model = initialized_depolarizing_noise(noise_level=noise_level)
        
        # Create an empty noise model
        noise_model = NoiseModel()

        # Add depolarizing error to all single qubit u1, u2, u3 gates
        error = depolarizing_error(noise_level, 1)
        noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3', 'rz', 'sx', 'x'])
        #normalize cnot error has 5% more percent noise than one-qubit gate
        cnot_error = depolarizing_error(noise_level * 5, 2)
        noise_model.add_all_qubit_quantum_error(cnot_error, ['cx'])

    elif noise_model.lower() == "thermal":
        num_physical_qubits = len(cmap.physical_qubits)
        noise_model = generate_noise_model(num_physical_qubits)
    else:
        noise_model = None

    job = execute(experiments=circuit,
                  backend=backend,
                  coupling_map=cmap,
                  noise_model= noise_model,
                  shots=1000,
                  initial_layout=init_layout,
                #   basis_gates=basis_gates,
                  optimization_level=optimization_level)
    
    return job.result().get_counts()


def generate_noise_model(n_qubit):
    # T1 and T2 values for qubits 0-3
    T1s = np.random.normal(50e3, 10e3, n_qubit) # Sampled from normal distribution mean 50 microsec
    T2s = np.random.normal(70e3, 10e3, n_qubit)  # Sampled from normal distribution mean 50 microsec

    # Truncate random T2s <= T1s
    T2s = np.array([min(T2s[j], 2 * T1s[j]) for j in range(n_qubit)])

    # Instruction times (in nanoseconds)
    time_u1 = 0   # virtual gate
    time_u2 = 50  # (single X90 pulse)
    time_u3 = 100 # (two X90 pulses)
    time_cx = 300
    time_reset = 1000  # 1 microsecond
    time_measure = 1000 # 1 microsecond

    # QuantumError objects
    errors_reset = [thermal_relaxation_error(t1, t2, time_reset)
                    for t1, t2 in zip(T1s, T2s)]
    errors_measure = [thermal_relaxation_error(t1, t2, time_measure)
                    for t1, t2 in zip(T1s, T2s)]
    errors_u1  = [thermal_relaxation_error(t1, t2, time_u1)
                for t1, t2 in zip(T1s, T2s)]
    errors_u2  = [thermal_relaxation_error(t1, t2, time_u2)
                for t1, t2 in zip(T1s, T2s)]
    errors_u3  = [thermal_relaxation_error(t1, t2, time_u3)
                for t1, t2 in zip(T1s, T2s)]
    errors_cx = [[thermal_relaxation_error(t1a, t2a, time_cx).expand(
                thermal_relaxation_error(t1b, t2b, time_cx))
                for t1a, t2a in zip(T1s, T2s)]
                for t1b, t2b in zip(T1s, T2s)]

    # Add errors to noise model
    noise_thermal = NoiseModel()
    for j in range(n_qubit):
        noise_thermal.add_quantum_error(errors_reset[j], "reset", [j])
        noise_thermal.add_quantum_error(errors_measure[j], "measure", [j])
        # noise_thermal.add_quantum_error(errors_u1[j], "rz", [j])
        # noise_thermal.add_quantum_error(errors_u2[j], "sx", [j])
        # noise_thermal.add_quantum_error(errors_u3[j], "x", [j])
        # noise_thermal.add_quantum_error(errors_u1[j], "u1", [j])
        # noise_thermal.add_quantum_error(errors_u2[j], "u2", [j])
        # noise_thermal.add_quantum_error(errors_u3[j], "u3", [j])
        for k in range(n_qubit):
            noise_thermal.add_quantum_error(errors_cx[j][k], "cx", [j, k])

    # print(noise_thermal)
    return noise_thermal