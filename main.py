
from qiskit import QuantumCircuit

from utils import *
from ibm_utils import *
from evaluate import * 


def run(qc, row, col, expected_result="", noise_level = 0, noise_model = "", optimization_level = 0):

    #convert qc to nxgraph
    qc_graph = qc_to_nxgraph(qc)

    #convert to max weighted path
    qc_spgraph = max_edges_spanning_tree(qc_graph)
    
    #generate grid matrix to find adjacency node
    grid_matrix = generate_grid_matrix(row, col, qc_spgraph)
    print(grid_matrix)

    #generated gird like coupling map graph with adjacency and diagonal edge 
    cmap_graph = connect_adjacent_and_diagonal_edge(grid_matrix, qc_spgraph, qc_graph)

    #eliminate the constraint of SQC
    cmap = eliminate_diagonal_edges_with_constraints(grid_matrix, cmap_graph)
    print(cmap.edges)

    print(f"Number of coupling: {len(cmap.edges)}")

    #Generate custom coupling map for Qiskit
    cmap = generate_custom_coupling_map(cmap)

    # evaluate_gate_count(qc, cmap, optimization_level=optimization_level)

    noisy_evaluation(qc, cmap, noise_level = noise_level, expected_result=expected_result, noise_model= noise_model, optimization_level=optimization_level)

if __name__ == "__main__":
    

    ### Use random circuit benchmark for SWAP reduction ### 
    # n_depth = 20
    # expected_result = ""
    # for i in range(10, 22):
    # # for i in range(10, 34):

    #     if i <= 20:
    #         row = 5
    #         col = 4
    #     else:
    #         row = 6
    #         col = 6
    #     for j in range(10):
    #         qasm_dir = f"tst/n{i}/rg_circuit_n{i}_{j}.qasm"
    #         qc = QuantumCircuit.from_qasm_file(qasm_dir)
    #         run(qc, row, col, optimization_level=0)

    ### Noisy benchcmark ###

    #result: good
    row = 2
    col = 5
    qasm_dir = "QASMBench/small/adder_n10/adder_n10_transpiled.qasm"
    expected_result = "10000"
    qc = QuantumCircuit.from_qasm_file(qasm_dir)

    #  #evaluate with many depolarizing noise
    error_rate = [0.0001, 0.0003, 0.0005, 0.0007, 0.001]
    for error_rate in error_rate:
        #evaluate 10 times for each error rate
        for i in range (0,10):
            print(f"######## Depolarizing Noise {error_rate} Evaluation ########")
            run(qc, row, col, expected_result=expected_result, noise_model="depolarizing", noise_level=error_rate, optimization_level=0)

    row = 4
    col = 5
    qasm_dir = "QASMBench/medium/bigadder_n18/bigadder_n18_transpiled.qasm"
    expected_result = "0 11000000"
    qc = QuantumCircuit.from_qasm_file(qasm_dir)

     #evaluate with many depolarizing noise
    error_rate = [0.0001, 0.0003, 0.0005, 0.0007, 0.001]

    # # error_rate = [0.001, 0.003, 0.005, 0.008, 0.01]
    for error_rate in error_rate:
        #evaluate 10 times for each error rate
        for i in range (0,10):
            print(f"######## Depolarizing Noise {error_rate} Evaluation ########")
            run(qc, row, col, expected_result=expected_result, noise_model="depolarizing", noise_level=error_rate, optimization_level=0)

    row = 4
    col = 4
    qasm_dir = f"QASMBench/medium/multiplier_n15/multiplier_n15_transpiled.qasm"
    expected_result = "001"

    #test with thermal relaxation
    qc = QuantumCircuit.from_qasm_file(qasm_dir)

    #evaluate with many depolarizing noise
    # error_rate = [0.0001, 0.0003, 0.0005, 0.0007, 0.001]
    for error_rate in error_rate:
        #evaluate 10 times for each error rate
        for i in range (0,10):
            print(f"######## Depolarizing Noise {error_rate} Evaluation ########")
            run(qc, row, col, expected_result=expected_result, noise_model="depolarizing", noise_level=error_rate, optimization_level=0)