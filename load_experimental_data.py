# import pickle
# import qiskit
#
# from qiskit import QuantumCircuit, transpile, IBMQ
# from copy import deepcopy
# from qiskit import Aer
# from qiskit.transpiler import PassManager
# from qiskit.transpiler.passes import Unroller
# from qiskit.transpiler.passes import BasisTranslator
# from qiskit.dagcircuit import DAGCircuit
# from qiskit.converters import circuit_to_dag
# # from qiskit.test.mock import FakeManhattan
# from qiskit.providers.fake_provider import FakeMontreal
# from qiskit.providers.models import BackendConfiguration
# from qiskit.transpiler import PassManager
# from qiskit import execute, Aer
# from qiskit_aer import QasmSimulator
# from qiskit_aer.noise import NoiseModel
# from qiskit.visualization import plot_histogram
# import time
# import pickle
# import datetime
# import os, argparse
# from qiskit import qpy
# from qiskit_ibm_runtime import QiskitRuntimeService, Session, Options, Sampler
from bitstring import BitArray
import itertools
import json
import relative_entropy_optimization
filepath1 = "./retrimqcresults/como_original_circuit_ibm_algiers_result_20240208_183720GHZ_13.json"
filepath2 = "./retrimqcresults/como_par+trim_circuit_ibm_algiers_result_20240208_183720GHZ_13.json"

def load_json_file(filepath):
    with open(filepath, 'r') as json_file:
        data = json.load(json_file)
    return data

# Usage
num_full_measurement_shots  = 4096
num_partial_measurements_shots = 315
n=13
N=2**n


def pad_bitstring(bitstring):#adds 0's in front of the bitstrings.
    return '0'*(13-len(bitstring)) + bitstring

def bitstring_to_int(bitstring, n=13):#Interprets the bitstring as binary and returns the associated integer
    #return BitArray(bin=bitstring).int + 2**(n-1)
    return int(bitstring, 2)

def bitstrings():
    return [''.join(i) for i in itertools.product('01', repeat=13)]


def test_bitstring_operations():
    bitstring = '001'
    print(bitstring_to_int(bitstring))
    padded =pad_bitstring(bitstring)
    print(padded)
    print(bitstring_to_int(bitstring))

    for x in bitstrings():
        print(x)
#test_bitstring_operations()


def partial_bitstring_to_subset(partial_bitstring, partial_bits):
    partial_bitstring = pad_bitstring(partial_bitstring) #make sure the bitstring is length 13.
    #print(partial_bits)
    #print(len(frozenset({bitstring_to_int(bs) for bs in bitstrings() if all([bs[i]==partial_bitstring[i] for i in partial_bits])})))
    # print(partial_bits)
    # print([partial_bitstring[12-i] for i in partial_bits])
    return {bitstring_to_int(bs) for bs in bitstrings() if all([bs[12-i]==partial_bitstring[12-i] for i in partial_bits])}

def ideal_partial_measurements(key,partial_bits):
    key=pad_bitstring(key)
    if len([k for k in key if k=='1'])%2==0:
        print('1')
        return 0.499
    else:
        print('0')
        return 0.001

def ideal_distribution():
    return [ 0.5 if b in ["0"*13, "1"*13] else 0 for b in bitstrings()]

def create_partial_meas_tup(pmeas, partial_bits):
    #print(pmeas)
    return [ (partial_bitstring_to_subset(key, partial_bits), max(0,pmeas[key]*num_partial_measurements_shots)) for key in pmeas.keys()]
    #return [ (partial_bitstring_to_subset(key, partial_bits), max(0,ideal_partial_measurements(key,partial_bits)*num_partial_measurements_shots)) for key in pmeas.keys()]
def create_ideal_partial_meas_tup(pmeas, partial_bits):
    return [ (partial_bitstring_to_subset(key, partial_bits), 0.499*num_partial_measurements_shots) if len([k for k in key if k=="1" ])%2==0 else (partial_bitstring_to_subset(key, partial_bits), 0.001*num_partial_measurements_shots) for key in pmeas.keys()]

def create_full_measurement_tuples(full_meas_option):
    if full_meas_option== "0":
        full_measurements = load_json_file(filepath1)[0] #A dictionary whose keys are bitstrings (representing the outcomes) and whose values are floats (representing the probability of the outcome).
    elif full_meas_option=="1":
        full_measurements = load_json_file(filepath1)[1] #A dictionary whose keys are bitstrings (representing the outcomes) and whose values are floats (representing the probability of the outcome).
    elif full_meas_option=="2":
        return None #todo
    full_measurements = {pad_bitstring(key) : full_measurements[key] for key in full_measurements.keys()}
    #print(type(full_measurements['0000000000000']))
    def get_value(outcome):
        if outcome in full_measurements.keys():
            return  max(full_measurements[outcome]* num_full_measurement_shots,0)
        else:
            return 0
    to_return = [ ({bitstring_to_int(outcome)}, get_value(outcome)) for outcome in bitstrings()]
    #to_return.sort(key = lambda x: list(x[0])[0])
    #to_return = relative_entropy_optimization.remove_redundancies(to_return)
    return to_return
    #return [( {bitstring_to_int(outcome)}, max(int(full_measurements[outcome]* num_full_measurement_shots),0)) for outcome in full_measurements.keys()]
#create_full_measurement_tuples()

def create_partial_measurement_tuples(p_meas_option):
    if p_meas_option == "untrimmed":
        partial_measurements = load_json_file(filepath2) #A list of 13 dictionaries, one for each partial measurement.
        return list(itertools.chain.from_iterable([create_partial_meas_tup(p_meas, [index, (index+1)%13 ]) for index, p_meas in enumerate(partial_measurements[0:13])]))
    elif p_meas_option == "trimmed":
        partial_measurements = load_json_file(filepath2) #A list of 13 dictionaries, one for each partial measurement.
        return list(itertools.chain.from_iterable([create_partial_meas_tup(p_meas, [index, (index+1)%13 ]) for index, p_meas in enumerate(partial_measurements[13:26])]))
    elif p_meas_option == "combined":
        return None #todo
    elif p_meas_option == "ideal":
        partial_measurements = load_json_file(filepath2)
        return list(itertools.chain.from_iterable([create_ideal_partial_meas_tup(p_meas, [index, (index+1)%13 ]) for index, p_meas in enumerate(partial_measurements[13:26])]))


def create_total_measurement_tuples(full_meas_option,p_meas_option):
    to_return = create_full_measurement_tuples(full_meas_option)+ create_partial_measurement_tuples(p_meas_option)
    return to_return

def success_probability(ans):
    ans = [(i,a) for i,a in enumerate(ans)]
    ans.sort(reverse=True, key= lambda x: x[1])
    assert {ans[0][0],ans[1][0]}=={bitstring_to_int("1"*13), bitstring_to_int("0"*13)}
    return ans[0][1]+ans[1][1]

def build_table_max_likelihood_row(full_meas_option,p_meas_option):
    totmeas=create_total_measurement_tuples(full_meas_option,p_meas_option)
    ans = list(relative_entropy_optimization.optimal_distribution(totmeas)[0:N])
    #ans.sort()
    ideal = ideal_distribution()
    #return relative_entropy_optimization.total_variation_distance(ans, ideal)
    return relative_entropy_optimization.hellinger_distance(ans,ideal)
    #return success_probability(ans)

def build_table_basic_row(full_meas_option,p_meas_option):
    fmeas = create_full_measurement_tuples(full_meas_option)
    #print(sum([f[1] for f in fmeas if f[1]<0]))
    total = sum([f[1] for f in fmeas])
    basic_dist = [ f[1]/ total  for f in fmeas]
    ideal = ideal_distribution()
    #return success_probability(ans)
    #return relative_entropy_optimization.total_variation_distance(basic_dist, ideal)
    return relative_entropy_optimization.hellinger_distance(basic_dist,ideal)

def build_table_jigsaw_row(full_meas_option,p_meas_option):
    totmeas=create_total_measurement_tuples(full_meas_option,p_meas_option)
    ans = relative_entropy_optimization.jigsaw_reconstruction(totmeas,N,alpha=1)
    ideal = ideal_distribution()
    #return success_probability(ans.p_func)
    #return relative_entropy_optimization.total_variation_distance(ans.p_func, ideal)
    return relative_entropy_optimization.hellinger_distance(ans.p_func,ideal)

def build_table_jigsaw_mod_row(full_meas_option,p_meas_option):
    totmeas=create_total_measurement_tuples(full_meas_option,p_meas_option)
    ans = relative_entropy_optimization.jigsaw_reconstruction(totmeas,N)
    #return success_probability(ans.p_func)
    ideal = ideal_distribution()
    #return relative_entropy_optimization.total_variation_distance(ans.p_func, ideal)
    return relative_entropy_optimization.hellinger_distance(ans.p_func,ideal)

def build_table_compressed_sensing_row(full_meas_option,p_meas_option):
    pmeas=create_partial_measurement_tuples(p_meas_option)
    ans = relative_entropy_optimization.compressed_sensing_dist(pmeas,N)
    ideal = ideal_distribution()
    return relative_entropy_optimization.hellinger_distance(ans,ideal)

def build_table_of_methods():

    full_meas_options,p_meas_options = ["0", "1"], ["untrimmed", "trimmed","ideal"]

    #methods = [build_table_basic_row, build_table_max_likelihood_row,build_table_jigsaw_mod_row, build_table_jigsaw_row]
    methods = [build_table_jigsaw_mod_row, build_table_jigsaw_row, build_table_compressed_sensing_row]
    for method in methods:
        for full_meas_option, p_meas_option in itertools.product(full_meas_options,p_meas_options):
            print(full_meas_option, p_meas_option, str(method.__name__), method(full_meas_option,p_meas_option))
build_table_of_methods()

    # jigsaw_ans = relative_entropy_optimization.jigsaw_reconstruction(totmeas,N,alpha=1)
    # jigsaw_ans.sort(reverse=True)
    # print('jigsaw', jigsaw_ans[0:3])
    # jigsaw_modified_ans = relative_entropy_optimization.jigsaw_reconstruction(totmeas,N)
    # jigsaw_modified_ans.sort(reverse=True)
    # print('jigsaw_mod', jigsaw_mo

#print(build_table_compressed_sensing_row("0", "untrimmed"))


def check_bitstring_to_int():
    for b in bitstrings():
        print(bitstring_to_int(b)) #should count 0,1,2,... 2^N-1
#check_bitstring_to_int()


# Method |Weigh partial_meas? | Full Meas | Partial Meas | Converged? | TVD |

#Methods: basic_ans
#         max likelyhood
#         Jigsaw basic
#         Jigsaw modified
