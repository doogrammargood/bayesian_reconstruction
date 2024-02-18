from cvxopt import solvers, matrix, spdiag, log, mul,div
import numpy as np
from itertools import chain, combinations, product, groupby
import bayesian_reconstruction
import matplotlib.pyplot as plt


def observation_matrix(samples_list): #makes sure that none of the samples occurs exactly 0 times.
    eps=0.000001
    #expects a list of pairs [(event , num_observations)]
    total = sum([pair[1]+eps for pair in samples_list])
    return matrix([(pair[1]+eps)/total for pair in samples_list] , (len(samples_list),1) )

def generate_partial_measurements(lst, n):#https://stackoverflow.com/questions/5360220/how-to-split-a-list-into-pairs-in-all-possible-ways
    #Returns a list of lists of sets. Each inner list is a partition of lst, and each element of the partition is a set of size n.
    #print("gpm", lst, n)
    if n ==1:
        yield [{l} for l in list(lst)]

    else:
        if not len(lst)%n==0:
            raise ValueError("length of the list should be divisible by n.")
        if not lst:
            yield []
        else:
            for group in (( {lst[0],}.union(set(xs))) for xs in combinations(lst[1:], n-1)):
                for groups in generate_partial_measurements([x for x in lst if x not in group], n):
                    yield [group] + groups
#print(list(generate_partial_measurements(range(6), 2)))


def prepare_constraint_matrix(samples_list):
    cons=[ [ 1.0 if pair_one[0].issubset(pair_two[0]) and len(pair_one[0])<len(pair_two[0]) and len(pair_one[0])==1 else -1.0 if pair_one[0]==pair_two[0] else 0.0 for pair_one in samples_list] for pair_two in samples_list if len(pair_two[0])>1]
    probability_sum = [ 1.0 if len(pair[0])==1 else 0. for pair in samples_list]
    #print("prob sum", sum(probability_sum))
    cons = [probability_sum]+cons
    cons = list(map(list, zip(*cons)))
    #print("contsraint matrix", matrix( list(chain.from_iterable(cons)),(len(cons[0]),len(samples_list)),'d'))
    return matrix( list(chain.from_iterable(cons)),(len(cons[0]),len(samples_list)),'d')
def prepare_contraint_rhs(samples_list):
    num_partial_measurements = len([s for s in samples_list if len(s[0])>1])
    rhs = [1.0 if i ==0 else 0.0 for i in range(num_partial_measurements+1)]
    #print("contsraints rhs",num_partial_measurements, rhs)
    return matrix(rhs,(num_partial_measurements+1,1),'d')

def optimal_distribution(samples_list):
    A= prepare_constraint_matrix(samples_list)
    b= prepare_contraint_rhs(samples_list)
    y = observation_matrix(samples_list)

    m, n = A.size
    def F(x=None, z=None):
        if x is None: return 0, matrix(1.0, (n,1))
        if min(x) <= 0.0: return None
        f=-sum(mul(y,log(x)))
        Df = -(mul(y,x**-1)).T
        if z is None: return f, Df
        H = spdiag(z[0] * mul(y, x**-2))
        return f, Df, H
    solvers.options['show_progress']=False
    #solvers.options['verbose']=True
    solvers.options['maxiters'] = 5000
    return solvers.cp(F, A=A, b=b)['x']

def simple_example():
    samples_list = [({0}, 2),({1} , 0),({2},6),({0,1}, 7)]
    ans = optimal_distribution(samples_list)
    print(ans)

def set_of_outcomes_to_integer(s,N):
    l = sorted(list(s),reverse=True)
    return sum([N**(len(l)-1-index) * item for index ,item  in enumerate(l) ])
#print(set_of_outcomes_to_integer({1,3,2},5))
def prepare_sample_list(dist,N,num_samples,singletons=False, noise = 0):
    samples = dist.sample(size = num_samples, noise = noise)
    samples_dict = {}
    if singletons:
        samples_dict = { s:0 for s in range(N) }
    for s in samples:
        if s in samples_dict.keys():
            samples_dict[s] = samples_dict[s]+1
        else:
            samples_dict[s] = 1
    samples_list = [ ( {s},samples_dict[s]) for s in samples_dict.keys()]
    def sort_function(s):
        return set_of_outcomes_to_integer(s[0],N)
    samples_list.sort(key = sort_function)
    return samples_list

def coarse_grain(samples_list, partition):
    to_return = []
    for p in partition:
        to_return = to_return + [(p, sum([sample[1] for sample in samples_list if sample[0].issubset(p)]) )]
    return to_return
# def remove_redundancies(samples_list):
#     to_return = []
#     seen = []
#     for p in samples_list:
#         if not p[0] in seen:
#             to_return.extend( [(p[0], sum([sample[1] for sample in samples_list if sample[0] == p[0] ]) )] )
#             seen.append(p[0])
#     return to_return
def remove_redundancies(samples_list):
    #print(sorted(samples_list, key=lambda x: x[0]))
    grouped_list = [list(g) for k, g in groupby(sorted(samples_list, key=lambda x: hash(frozenset(x[0]))), lambda x: x[0])] #https://stackoverflow.com/questions/6602172/how-to-group-a-list-of-tuples-objects-by-similar-index-attribute-in-python
    reduced_list = [ (g[0][0], sum([h[1] for h in g])) for g in grouped_list]
    return reduced_list

def test_remove_redundancies():
    samples_list = [({0}, 2),({1} , 0),({2},6),({0,1}, 7), ({1,0}, 5), ({2,0},3), ({1,0}, 4), ({0},4)]
    print(remove_redundancies(samples_list))
#test_remove_redundancies()


def total_variation_distance(list_1,list_2):
    assert(len(list_1)==len(list_2))
    total = 0
    for index in range(len(list_1)):
        total=total+ abs(list_1[index]-list_2[index])
    return total

def hellinger_distance(P,Q):

    assert(len(P)==len(Q))
    return 2**(-0.5) *(sum([ (P[i]**(0.5) - Q[i]**(0.5) )**2 for i in range(len(P))]))**(0.5)

def random_example():
    N=10
    M=3
    dist = bayesian_reconstruction.ProbabilityDistribution(N, bayesian_reconstruction.uniform_sample_simplex(N))
    samples_list = prepare_sample_list(dist,N,1000,singletons=True)
    outcome_partition1 = [{0,1},{2,3},{4,5},{6,7},{8,9}]
    partial_samples_list1 = prepare_sample_list(dist,N,10000)
    outcome_partition2 = [{0,9},{1,8},{2,7},{3,6},{4,5}]
    partial_samples_list2 = prepare_sample_list(dist,N,10000)
    outcome_partition3 = [{2,8},{1,3},{0,6},{5,7},{4,9}]
    partial_samples_list3 = prepare_sample_list(dist,N,10000)

    big_samples_list =[]
    samples_lists = [samples_list,coarse_grain(partial_samples_list1,outcome_partition1 ),coarse_grain(partial_samples_list2,outcome_partition2 ),coarse_grain(partial_samples_list3,outcome_partition3 ) ]
    for l in samples_lists:
        big_samples_list.extend(l)
    big_samples_list= remove_redundancies(big_samples_list)
    print(big_samples_list)

    print(dist.p_func)
    print(samples_list)
    basic_answer = optimal_distribution(samples_list)[0:N]
    corrected_answer = optimal_distribution(big_samples_list)[0:N]
    averages = [pair[1]/1000 for pair in samples_list]
    print(total_variation_distance(dist.p_func, basic_answer))
    print(total_variation_distance(dist.p_func,corrected_answer))
#random_example()

def simulate_partial_measurements(dist,partial_measurement_list, N,sample_size=1000, noise = 0):
    partial_samples_list = []
    for partition in partial_measurement_list:
        current_partial_samples= prepare_sample_list(dist,N,sample_size,noise=noise)
        partial_samples_list.extend(coarse_grain(current_partial_samples, partition ) )
    return(remove_redundancies(partial_samples_list))
def complete_example():
    N=10
    dist = bayesian_reconstruction.ProbabilityDistribution(N, bayesian_reconstruction.uniform_sample_simplex(N))
    pmeas = simulate_partial_measurements(dist, generate_partial_measurements(range(N), 2), N,sample_size=10000)
    fmeas = simulate_partial_measurements(dist, generate_partial_measurements(range(N), 1), N)
    #print(fmeas)
    totmeas = pmeas + fmeas
    def sort_function(s):
        return set_of_outcomes_to_integer(s[0],N)
    totmeas.sort(key = sort_function)
    print("tot,part,full", totmeas, pmeas,fmeas)
    print("distfunc", dist.p_func)
    basic_answer = optimal_distribution(fmeas)[0:N]
    print("basic_answer dev",total_variation_distance(dist.p_func, basic_answer))
    corrected_answer = optimal_distribution(totmeas)[0:N]
    print("corrected answer dev",total_variation_distance(dist.p_func, corrected_answer))

def jigsaw_new_samples(current_distribution,pmeas,N,alpha):
    new_distribution = []
    for p in pmeas:
        marginal = current_distribution.marginal(p[0])
        l = list(p[0])
        l.sort()
        tot = sum([current_distribution.p_func[o] for o in l])
        for index, x in enumerate(l):
            if alpha == 0:
                new_distribution.append(({x},p[1]*marginal.p_func[index] ) )
            elif alpha == 1:
                new_distribution.append(({x},p[1]*marginal.p_func[index]/((1.0 - tot)) ) )
            else:
                new_distribution.append(({x},p[1]*marginal.p_func[index]/((1.0 - tot)**alpha) ) )
    new_distribution = remove_redundancies(new_distribution)
    new_distribution = [ pair[1] for pair in new_distribution]
    total = sum(new_distribution)
    new_distribution = [ n/total for n in new_distribution]
    new_distribution = bayesian_reconstruction.ProbabilityDistribution(N, p_func = new_distribution)
    return new_distribution

def jigsaw_reconstruction(samples_list,N,t=0, alpha =0, num_iterations=10 ):
    #implements the technique from the jigsaw paper to reconstruct the distribution from samples_list.
    fmeas = [(s[0], s[1]) for s in samples_list if len(s[0])==1]
    pmeas = [(s[0], s[1]) for s in samples_list if len(s[0])>1]
    denom = sum([s[1] for s in fmeas])
    current_distribution = bayesian_reconstruction.ProbabilityDistribution(N, p_func=[ s[1]/denom for s in fmeas])
    count = 0
    while count <num_iterations:
        new_distribution = jigsaw_new_samples(current_distribution,pmeas,N ,alpha)
        if hellinger_distance(current_distribution.p_func, new_distribution.p_func) < .0001:
            print("jigsaw completed")
            return new_distribution
        current_distribution=current_distribution.average_with(new_distribution,t)
        count = count+1
    #print("jigsaw timed out.")
    return(current_distribution)

def jigsaw_example():
    N=10
    dist = bayesian_reconstruction.ProbabilityDistribution(N, bayesian_reconstruction.uniform_sample_simplex(N))
    random_to_compare = bayesian_reconstruction.ProbabilityDistribution(N, bayesian_reconstruction.uniform_sample_simplex(N))
    pmeas = simulate_partial_measurements(dist, generate_partial_measurements(range(N), 2), N,sample_size=100000)
    fmeas = simulate_partial_measurements(dist, generate_partial_measurements(range(N), 1), N, sample_size = 10000)
    #print(fmeas)
    totmeas = pmeas + fmeas
    def sort_function(s):
        return set_of_outcomes_to_integer(s[0],N)
    totmeas.sort(key = sort_function)
    jigsaw_ans = jigsaw_reconstruction(totmeas,N,alpha=1)
    jigsaw_modified_ans = jigsaw_reconstruction(totmeas,N)
    basic_answer = optimal_distribution(fmeas)[0:N]
    corrected_answer = optimal_distribution(totmeas)[0:N]
    print("basic", total_variation_distance(dist.p_func, basic_answer))
    print("jigsaw", total_variation_distance(dist.p_func, jigsaw_ans.p_func))
    print("jigsaw modified", total_variation_distance(dist.p_func, jigsaw_modified_ans.p_func))
    print("corrected", total_variation_distance(dist.p_func, corrected_answer))
    print("random", total_variation_distance(dist.p_func,random_to_compare.p_func))
    return jigsaw_ans
#jigsaw_example()

def bitlist_to_int(bitlist):
    binary_str = ''.join(map(str, bitlist))
    decimal_int = int(binary_str, 2)
    return decimal_int

def bitlist_index(bitlist, indices):
    return [ b for i, b in enumerate(bitlist) if i in indices]

def partition_from_qubits(qubits_to_be_measured,n):
    #expect qubits_to_be_measured to be a list of qubits.
    m=len(qubits_to_be_measured)
    measured_bits= list(product([0,1], repeat = m))
    total_bits = list(product([0,1], repeat=n))
    to_return = [ { bitlist_to_int(tot) for tot in total_bits if bitlist_index(tot,qubits_to_be_measured)==list(out)} for out in measured_bits]
    return to_return

def p_meas_qubits(qubits_list,n):
    return [partition_from_qubits(meas,n) for meas in qubits_list]
#partition_from_qubits([2],3)

def sampler(n,qubit_list,dist = None,pmeas_sample_size = 1000):
    #n is the number of qubits.
    #qubits list is a list of tuples. Each tuple is a set of qubits to measure.
    N= 2**n
    if dist == None:
        dist = bayesian_reconstruction.ProbabilityDistribution(N, bayesian_reconstruction.uniform_sample_simplex(N))
    fmeas = simulate_partial_measurements(dist, generate_partial_measurements(range(N), 1), N, sample_size = 1000, noise =0)
    pmeas = simulate_partial_measurements(dist, p_meas_qubits(qubit_list,n),N, sample_size = pmeas_sample_size, noise =0)
    totmeas = pmeas + fmeas
    def sort_function(s):
        return set_of_outcomes_to_integer(s[0],N)
    totmeas.sort(key = sort_function)
    return totmeas,fmeas,pmeas

def qubit_partial_measurement_example(n):
    N= 2**n
    #qubit_list = list(combinations(range(n),2))
    qubit_list = [(index%n, (index+1)%n) for index in range(n)]

    dist = bayesian_reconstruction.ProbabilityDistribution(N, bayesian_reconstruction.uniform_sample_simplex(N))
    totmeas,fmeas,pmeas = sampler(n,qubit_list,dist=dist)
    jigsaw_ans = jigsaw_reconstruction(totmeas,N,alpha=1)
    jigsaw_modified_ans = jigsaw_reconstruction(totmeas,N)
    basic_answer = optimal_distribution(fmeas)[0:N]
    corrected_answer = optimal_distribution(totmeas)[0:N]
    print("basic", total_variation_distance(dist.p_func, basic_answer))
    print("jigsaw", total_variation_distance(dist.p_func, jigsaw_ans.p_func))
    print("jigsaw modified", total_variation_distance(dist.p_func, jigsaw_modified_ans.p_func))
    print("corrected", total_variation_distance(dist.p_func, corrected_answer))

def generate_noiseless_plot(n):
    N= 2**n
    num_trials = 1000
    #pmeas_sample_sizes = [100,500,1000,5000,10000,50000,100000]
    pmeas_sample_sizes = [2**k for k in [3,4,5,6,7,8,9,10,11,12]]
    #pmeas_sample_sizes = [100,500]

    basic_answers=[]
    jigsaw_answers=[]
    jigsaw_modified_answers=[]
    corrected_answers=[]
    t= 1 / (n*(n+1)/2+1)
    for trials in range(num_trials):
        total_answers = []
        dist = bayesian_reconstruction.ProbabilityDistribution(N, bayesian_reconstruction.uniform_sample_simplex(N))

        basic_answer_tv=[]
        jigsaw_ans_tv=[]
        jigsaw_modified_ans_tv=[]
        corrected_answer_tv=[]

        for pmeas_sample_size in pmeas_sample_sizes:

            qubit_list = list(combinations(range(n),2))
            totmeas,fmeas,pmeas = sampler(n,qubit_list,dist=dist,pmeas_sample_size=pmeas_sample_size)
            jigsaw_ans = jigsaw_reconstruction(totmeas,N,t=t,alpha=1)
            jigsaw_modified_ans = jigsaw_reconstruction(totmeas,N,t=t)
            basic_answer = optimal_distribution(fmeas)[0:N]
            corrected_answer= optimal_distribution(totmeas)[0:N]

            basic_answer_tv.append(total_variation_distance(dist.p_func, basic_answer))
            jigsaw_ans_tv.append(total_variation_distance(dist.p_func, jigsaw_ans.p_func))
            jigsaw_modified_ans_tv.append(total_variation_distance(dist.p_func, jigsaw_modified_ans.p_func))
            corrected_answer_tv.append(total_variation_distance(dist.p_func, corrected_answer))

        basic_answers.append(basic_answer_tv)
        jigsaw_answers.append(jigsaw_ans_tv)
        jigsaw_modified_answers.append(jigsaw_modified_ans_tv)
        corrected_answers.append(corrected_answer_tv)

    basic_answer_tv = [[ row[k] for row in basic_answers ] for k in range(len(pmeas_sample_sizes))]
    jigsaw_ans_tv = [[ row[k] for row in jigsaw_answers ] for k in range(len(pmeas_sample_sizes))]
    jigsaw_modified_ans_tv = [[ row[k] for row in jigsaw_modified_answers ] for k in range(len(pmeas_sample_sizes))]
    corrected_answer_tv = [[ row[k] for row in corrected_answers ] for k in range(len(pmeas_sample_sizes))]

    plt.figure(figsize=(8, 6))
    plt.errorbar(pmeas_sample_sizes, [np.mean(tv) for tv in basic_answer_tv], yerr=[np.var(tv)**0.5 for tv in basic_answer_tv], label='Full measurements only', fmt='o', color = 'black')
    plt.errorbar(pmeas_sample_sizes, [np.mean(tv) for tv in jigsaw_ans_tv], yerr =[np.var(tv)**0.5 for tv in jigsaw_ans_tv], label = 'Original Jigsaw', fmt ='o', color = 'green')
    plt.errorbar(pmeas_sample_sizes, [np.mean(tv) for tv in jigsaw_modified_ans_tv], yerr= [np.var(tv)**0.5 for tv in jigsaw_modified_ans_tv], label = 'Modified Jigsaw', fmt='o', color = 'blue')
    plt.errorbar(pmeas_sample_sizes, [np.mean(tv) for tv in corrected_answer_tv], yerr=[np.var(tv)**0.5 for tv in corrected_answer_tv], label = 'Optimal reconstruction', fmt='o', color = 'red')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Log of partial measurement samples')
    plt.ylabel('Log of total variation distance to true distribution')
    plt.title("Comparison of reconstruction methods for " + str(n) + " qubits")
    plt.legend()
    plt.savefig('{}_qubits_plot.png'.format(n))
    plt.show()

def check_scalability():
    for n in range(5,14):
        qubit_partial_measurement_example(n)

#check_scalability()
#generate_noiseless_plot(7)

#qubit_partial_measurement_example(5)
#print(set_of_outcomes_to_integer({0},10), set_of_outcomes_to_integer({0,1},10))
