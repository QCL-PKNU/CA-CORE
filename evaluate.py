

import qiskit_aer as Aer

from qiskit import execute, transpile
from qiskit.providers.fake_provider import FakeAlmadenV2, FakeCairoV2, FakeJohannesburgV2, FakePrague, FakeTokyo, ConfigurableFakeBackend

from utils import *
from ibm_utils import *


def noisy_evaluation(qc, cmap, noise_level, expected_result, noise_model, optimization_level=0):

    shots = 1000
    log = "\n"

    #No noise simulation
    exact_result = ibm_executor(circuit=qc.copy(),
                                cmap=None,
                                optimization_level=optimization_level,
                                noise_level=0.0,
                                init_layout=None)
    
    print(f"Exact Result: {exact_result}")
    
    log += f"{noise_level},{exact_result.get(expected_result)/shots},"

    #Generate configrable fakebackend
    n_qubits = len(cmap.physical_qubits)
    ReT_backend = ConfigurableFakeBackend("ReT",
                                          n_qubits=n_qubits,
                                          coupling_map=list(cmap.get_edges()),
                                          version=1)

    #ReT Simulation
    #TEST: ReT_backend
    ReT_result = ibm_executor(circuit=qc.copy(),
                                cmap=cmap,
                                backend=ReT_backend,
                                optimization_level=optimization_level,
                                noise_level=noise_level,
                                noise_model=noise_model,
                                init_layout=None)
    print(f"ReT Result {expected_result}: {ReT_result.get(expected_result)/shots}")

    #transpiled ReT_L = Reconfigurable Topology with initial layout
    init_layout = {}
    for i in range(len(qc.qubits)):
        init_layout[i] = qc.qubits[i]
    ReT_L_result = ibm_executor(circuit=qc.copy(),
                                cmap=cmap,
                                backend=ReT_backend,
                                optimization_level=optimization_level,
                                noise_level=noise_level,
                                noise_model=noise_model,
                                init_layout=init_layout)
    print(f"ReT_L Result {expected_result}: {ReT_L_result.get(expected_result)/shots}")
    log += f"{ReT_L_result.get(expected_result)/shots},"
    

    #List of Fake backend with different topology
    fake_almaden = FakeAlmadenV2() #20-qubit
    fake_johannesburg = FakeJohannesburgV2() #20-qubit
    fake_cairo = FakeCairoV2() #27-qubit
    # fake_tokyo = FakeTokyo()

    #Fake Backend Evaluation
    for fake_backend in [fake_almaden, fake_johannesburg, fake_cairo]:
        if fake_backend.num_qubits < len(qc.qubits):
            print(f"Backend {fake_backend.name} does not have enough resources for execution {len(qc.qubits)}-Qubit")
            continue

        fake_backend_result = ibm_executor(circuit=qc.copy(),
                                cmap=fake_backend.coupling_map,
                                optimization_level=optimization_level,
                                noise_level=noise_level,
                                noise_model=noise_model,
                                backend=fake_backend,
                                init_layout=None)
        
        print(f"{fake_backend.name} Result {expected_result}: {fake_backend_result.get(expected_result)}")
        log += f"{fake_backend_result.get(expected_result)/shots},"
    
    #Generate fake sycamore 
    sycamore_graph = generate_sycamore_toplogy_graph(4, 6) #sycamore topology with 24-qubit
    sycamore_cmap = generate_custom_coupling_map(sycamore_graph)
    n_qubits = len(cmap.physical_qubits)
    sycamore_backend = ConfigurableFakeBackend("ReT",
                                          n_qubits=n_qubits,
                                          coupling_map=list(sycamore_cmap.get_edges()),
                                          version=1)
    sycamore_result = ibm_executor(circuit=qc.copy(),
                                cmap=sycamore_cmap,
                                backend=sycamore_backend,
                                optimization_level=optimization_level,
                                noise_level=noise_level,
                                noise_model=noise_model,
                                init_layout=None)
    print(f"Sycamore-24 qubits Result {expected_result}: {sycamore_result.get(expected_result)}")
    log += f"{sycamore_result.get(expected_result)/shots},"

    

    f = open("noisy_bench.csv", 'a')
    f.write(log)
    f.close()


def evaluate_gate_count(qc, cmap, optimization_level):

    #setup
    backend = Aer.get_backend("qasm_simulator")
    
    #List of Fake backend with different topology
    fake_almaden = FakeAlmadenV2() #20-qubit
    fake_johannesburg = FakeJohannesburgV2() #20-qubit
    fake_cairo = FakeCairoV2() #27-qubit
    fake_prague = FakePrague() #33-qubit

    # fake_backends = [fake_almaden, fake_johannesburg, fake_cairo, fake_prague, fake_sycamore]
    fake_backends = [fake_almaden, fake_cairo, fake_prague]


    #optimization level for qiskit transpiler
    # optimization_level = 0

    #use basis gate from qasm_simulator instead of fake backend
    basis_gates = backend.configuration().basis_gates
    # basis_gates = fake_almaden

    init_layout = {}
    for i in range(len(qc.qubits)):
        init_layout[i] = qc.qubits[i]

    #transpiled ReT: Reconfigurable Topology
    transpiled_ReT = transpile(
        circuits=qc.copy(),
        backend=backend,
        basis_gates=basis_gates,
        coupling_map=cmap,
        optimization_level=optimization_level
    )

    #transpiled ReT_L = Reconfigurable Topology with initial layout
    init_layout = {}
    for i in range(len(qc.qubits)):
        init_layout[i] = qc.qubits[i]
    
    #transpiled ReT: Reconfigurable Topology
    transpiled_ReT_L = transpile(
        circuits=qc.copy(),
        backend=backend,
        basis_gates=basis_gates,
        coupling_map=cmap,
        initial_layout = init_layout,
        optimization_level=optimization_level
    )

    #Log
    log = f"\n{qc.num_qubits},"

    print(f"############ {len(qc.qubits)}-Qubits ############")
    ####Original Circuit####
    print(f"Original Quantum Circuit Gate Count")
    print(qc.count_ops())
    #log
    depth = qc.depth()
    print("Depth:", depth)
    gates = sum(qc.count_ops().values())
    cxs = qc.count_ops()['cx']
    swaps = 0
    # log += f"Original Circuit, {depth}, {gates}, {cxs}, {swaps}\n"
    log += f"{depth}, {gates}, {swaps},"

    ####ReT####
    print("\nReT Quantum Circuit Gate Count")
    print(transpiled_ReT.count_ops())
    #log
    depth = transpiled_ReT.depth()
    print("Depth:", depth)
    gates = sum(transpiled_ReT.count_ops().values())
    cxs = transpiled_ReT.count_ops()['cx']
    swaps = transpiled_ReT.count_ops()['swap']
    # log += f"ReT, {depth}, {gates}, {cxs}, {swaps}\n"
    # log += f"{depth}, {gates}, {swaps},"


    ####ReT_L####
    print("\nReT_L Quantum Circuit Gate Count")
    print(transpiled_ReT_L.count_ops())
    #log
    depth = transpiled_ReT_L.depth()
    print("Depth:", depth)
    gates = sum(transpiled_ReT_L.count_ops().values())
    cxs = transpiled_ReT_L.count_ops()['cx']
    swaps = transpiled_ReT_L.count_ops()['swap']
    # log += f"ReT_L, {depth}, {gates}, {cxs}, {swaps}\n"
    log += f"{depth}, {gates}, {swaps},"

    ###IBM Tokyo###
    if len(qc.qubits) < 21:
        tokyo_graph = generate_ibm_tokyo_topology_graph()
        tokyo_cmap = generate_custom_coupling_map(tokyo_graph)
        tokyo_transpiled = transpile(
            circuits=qc.copy(),
            backend=backend,
            coupling_map=tokyo_cmap,
            basis_gates=basis_gates,
            optimization_level=optimization_level
        )
        print(f"\nTokyo Circuit Gate Count")
        print(tokyo_transpiled.count_ops())
        depth = tokyo_transpiled.depth()
        print("Depth:", depth)
        gates = sum(tokyo_transpiled.count_ops().values())
        cxs = tokyo_transpiled.count_ops()['cx']
        swaps = tokyo_transpiled.count_ops()['swap']
        log += f"{depth}, {gates}, {swaps},"
    else:
        log+=",,,"

    #Fake Backend Evaluation
    for fake_backend in fake_backends:
        if fake_backend.num_qubits < len(qc.qubits):
            print(f"Backend {fake_backend.name} does not have enough resources for executing {len(qc.qubits)}-Qubit")
            log += f",,,"

            continue

        transpiled = transpile(
            circuits=qc.copy(),
            backend=backend,
            # backend=fake_almaden,
            coupling_map=fake_backend.coupling_map,
            basis_gates=basis_gates,
            # noise_model = noise_model,
            optimization_level=optimization_level
        )
        
        print(f"\n{fake_backend.name} Circuit Gate Count")
        print(transpiled.count_ops())
        #log
        depth = transpiled.depth()
        print("Depth:", depth)
        gates = sum(transpiled.count_ops().values())
        cxs = transpiled.count_ops()['cx']
        swaps = transpiled.count_ops()['swap']
        log += f"{depth}, {gates}, {swaps},"
    
    #Custom Backend
    #generate sycamore fake backend
    #54-qubit
    sycamore_graph = generate_sycamore_toplogy_graph(row=9, col=6)
    sycamore_cmap = generate_custom_coupling_map(sycamore_graph)
    # fake_sycamore = ConfigurableFakeBackend("Sycamore", 
    #                                         len(sycamore_graph.nodes),
    #                                         coupling_map=list(sycamore_graph.edges),
    #                                         version=1)
    sycamore_transpiled = transpile(
        circuits=qc.copy(),
        backend=backend,
        coupling_map=sycamore_cmap,
        basis_gates=basis_gates,
        optimization_level=optimization_level
    )
    print(f"\nSycamore Circuit Gate Count")
    print(sycamore_transpiled.count_ops())
    depth = sycamore_transpiled.depth()
    print("Depth:", depth)
    gates = sum(sycamore_transpiled.count_ops().values())
    cxs = sycamore_transpiled.count_ops()['cx']
    swaps = sycamore_transpiled.count_ops()['swap']
    log += f"{depth}, {gates}, {swaps},"

    #log result
    # f = open(f"cx_bench{len(qc.qubits)}.csv", "a")
    f = open(f"swap_bench.csv", "a")
    f.write(log)
    f.close()