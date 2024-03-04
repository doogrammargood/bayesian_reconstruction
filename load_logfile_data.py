from bitstring import BitArray
import itertools
import sys
sys.path.insert(0, '/Users/victorbankston/Documents/papers')
import relative_entropy_optimization


filepath= "./ibmq_manhattan/ghz14/ghz14_ibmq_manhattan_partial_circ_len_2_0_20.txt"
#filepath= "./toy_example.txt"
# original_reconstruction_script_filepath = "../scripts"
# sys.path.insert('0', original_reconstruction_script_filepath)
#import Reconstruction_Functions as RF

n=14
#n=3
N=2**n
with open(filepath, 'r') as file:
    data = file.readlines()[1:]
#print(data[0])
exec(data[0],globals(), locals() )
for d in data:
    try:
        exec(d)
    except Exception as e:
        print(f"Error executing line: {d}")
        print(f"Error message: {e}")
        exec(d,globals(), locals() )
#
# print(partial_logical_qubits)
# print(baseline_counts_dict)
# print(pp_counts_dict)

def ideal_distribution_ghz(n):
    return [ 0.5 if b in ["0"*n, "1"*n] else 0 for b in bitstrings(n)]

def bitstrings(n):
    return [''.join(i) for i in itertools.product('01', repeat=n)]

def bitstring_to_int(bitstring, n=13):#Interprets the bitstring as binary and returns the associated integer
    return int(bitstring, 2)

def convert_to_samples_list(baseline_counts_dict, partial_logical_qubits, pp_counts_dict,n):

    full_samples_list = [({bitstring_to_int(s,n=n)}, baseline_counts_dict[s]) if s in baseline_counts_dict.keys() else ({bitstring_to_int(s,n=n)}, 0.0001)  for s in bitstrings(n)]
    assert(len(pp_counts_dict)==len(partial_logical_qubits))
    p_samples = zip(partial_logical_qubits, pp_counts_dict)
    partial_samples_list = []
    for qubits_to_measure, partial_samples in p_samples:
        #print("qubits", qubits_to_measure, "partialsamples  ", partial_samples[0])
        #print("loo",list(partial_samples[0].keys()))[0]
        partial_samples_list.extend([({ bitstring_to_int(s,n=n) for s in bitstrings(n)  if all([ s[qubit]==k[index] for index, qubit in enumerate(qubits_to_measure)] )}, partial_samples[0][k]) for k in partial_samples[0].keys() ])
        #print([({ bitstring_to_int(s,n=n) for s in bitstrings(n)  if all([ s[qubit]==k[index] for index, qubit in enumerate(qubits_to_measure)] )}, partial_samples[0][k]) for k in partial_samples[0].keys() ])
    #print(partial_samples_list)
    # for k in partial_samples_list:
    #     print(len(k[0]))
    return full_samples_list + partial_samples_list


samples_list = convert_to_samples_list(baseline_counts_dict[0], partial_logical_qubits, pp_counts_dict,n=n)
#print("samples list ", samples_list)
fmeas = [f for f in samples_list if len(f[0]) == 1]
total = sum([f[1] for f in fmeas])
basic_dist = [f[1]/ total  for f in fmeas]
#print(samples_list)
jigsaw_dist = relative_entropy_optimization.jigsaw_reconstruction(samples_list,N,alpha=1,num_iterations=10).p_func
jigsaw_modified_dist = relative_entropy_optimization.jigsaw_reconstruction(samples_list,N,alpha=0,num_iterations=10).p_func
ideal_dist = ideal_distribution_ghz(n)

#original_jigsaw_dist = RF.recurr_reconstruct(baseline_counts_dict[0], pp_counts_dict,max_recur_count)

#samples_list = relative_entropy_optimization.observation_matrix(samples_list)
max_likelihood_dist = relative_entropy_optimization.optimal_distribution(samples_list)[0:N]


print("basic TVD: ", relative_entropy_optimization.total_variation_distance(basic_dist, ideal_dist))
print("max likelihood TVD: ", relative_entropy_optimization.total_variation_distance(max_likelihood_dist, ideal_dist))
print("jigsaw TVD: ", relative_entropy_optimization.total_variation_distance(jigsaw_dist, ideal_dist))
print("ideal TVD: ", relative_entropy_optimization.total_variation_distance(ideal_dist, ideal_dist))

print("basic PST ", basic_dist[0]+ basic_dist[-1])
print("max likelihood PST ", max_likelihood_dist[0]+ max_likelihood_dist[-1])
print("jigsaw PST ", jigsaw_dist[0]+jigsaw_dist[-1])
print("jigsaw modified PST ", jigsaw_modified_dist[0]+ jigsaw_modified_dist[-1])
print("ideal PST ", ideal_dist[0]+ideal_dist[-1])
# print("jigsaw ", jigsaw_dist[0], jigsaw_dist[-1])
# jigsaw_dist.sort(reverse=True)
# print("max jigsaw entries ", jigsaw_dist[0],jigsaw_dist[1],jigsaw_dist[2])


#x = "partial_logical_qubits = [[1, 0], [2, 1], [3, 2], [4, 3], [5, 4], [6, 5], [7, 6], [8, 7], [9, 8], [10, 9], [11, 10], [12, 11], [13, 12], [13, 0]]"
#print(x)
#exec(x)
#partial_logical_qubits = [[1, 0], [2, 1], [3, 2], [4, 3], [5, 4], [6, 5], [7, 6], [8, 7], [9, 8], [10, 9], [11, 10], [12, 11], [13, 12], [13, 0]]
