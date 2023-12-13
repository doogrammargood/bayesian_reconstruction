from cvxopt import solvers, matrix, spdiag, log, mul,div
import numpy as np
from itertools import chain, combinations
import bayesian_reconstruction

def observation_matrix(samples_list):
    #expects a list of pairs [(event , num_observations)]
    total = sum([pair[1] for pair in samples_list])
    return matrix([pair[1]/total for pair in samples_list] , (len(samples_list),1) )

def generate_partial_measurements(lst, n):#https://stackoverflow.com/questions/5360220/how-to-split-a-list-into-pairs-in-all-possible-ways
    #Returns a list of lists of sets. Each inner list is a partition of lst, and each element of the partition is a set of size n.
    if not len(lst)%n==0:
        raise ValueError("length of the list should be divisible by n.")
    if not lst:
        yield []
    else:
        for group in (( {lst[0],} .union(set(xs))) for xs in combinations(lst[1:], n-1)):
            for groups in generate_partial_measurements([x for x in lst if x not in group], n):
                yield [group] + groups
#print(list(generate_partial_measurements(range(6), 2)))


def prepare_constraint_matrix(samples_list):
    cons=[ [ 1.0 if pair_one[0].issubset(pair_two[0]) and len(pair_one[0])<len(pair_two[0]) and len(pair_one[0])==1 else -1.0 if pair_one[0]==pair_two[0] else 0.0 for pair_one in samples_list] for pair_two in samples_list if len(pair_two[0])>1]
    probability_sum = [ 1.0 if len(pair[0])==1 else 0. for pair in samples_list]
    cons = [probability_sum]+cons
    cons = list(map(list, zip(*cons)))
    return matrix( list(chain.from_iterable(cons)),(len(cons[0]),len(samples_list)),'d')
def prepare_contraint_rhs(samples_list):
    num_partial_measurements = len([s for s in samples_list if len(s[0])>1])
    rhs = [1.0 if i ==0 else 0.0 for i in range(num_partial_measurements+1)]
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
    return solvers.cp(F, A=A, b=b)['x']

def simple_example():
    samples_list = [({0}, 2),({1} , 0),({2},6),({0,1}, 7)]
    ans = optimal_distribution(samples_list)
    print(ans)

def set_of_outcomes_to_integer(s,N):
    l = sorted(list(s),reverse=True)
    return sum([N**(len(l)-1-index) * item for index ,item  in enumerate(l) ])
#print(set_of_outcomes_to_integer({1,3,2},5))
def prepare_sample_list(dist,N,num_samples,singletons=False):
    samples = dist.sample(size = num_samples)
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
def remove_redundancies(samples_list):
    to_return = []
    seen = []
    for p in samples_list:
        if not p[0] in seen:
            to_return.extend( [(p[0], sum([sample[1] for sample in samples_list if sample[0] == p[0] ]) )] )
            seen.append(p[0])
    return to_return

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

def simulate_partial_measurements(dist,partial_measurement_list, N,sample_size=1000):
    partial_samples_list = []
    for partition in partial_measurement_list:
        current_partial_samples= prepare_sample_list(dist,N,sample_size)
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
#complete_example()

def jigsaw_reconstruction(samples_list,N, alpha =0 ):
    #implements the technique from the jigsaw paper to reconstruct the distribution from samples_list.
    fmeas = [(s[0], s[1]) for s in samples_list if len(s[0])==1]
    pmeas = [(s[0], s[1]) for s in samples_list if len(s[0])>1]
    denom = sum([s[1] for s in fmeas])
    #print("fmeas", fmeas)#
    current_distribution = bayesian_reconstruction.ProbabilityDistribution(N, p_func=[ s[1]/denom for s in fmeas])

    while True:
        new_distribution = []
        # print(pmeas)
        # print(current_distribution.p_func)
        for p in pmeas:
            # print("p",p)
            marginal = current_distribution.marginal(p[0])
            l = list(p[0])
            l.sort()
            tot = sum([current_distribution.p_func[o] for o in l])
            for index, x in enumerate(l):
                #print(index,x)
                if alpha == 0:
                    new_distribution.append(({x},p[1]*marginal.p_func[index] ) )
                elif alpha == 1:
                    new_distribution.append(({x},p[1]*marginal.p_func[index]/((1.0 - tot)) ) )
                else:
                    new_distribution.append(({x},p[1]*marginal.p_func[index]/((1.0 - tot)**alpha) ) )
                #print(({x},p[1]*marginal.p_func[index] ))
        #print("new_dist",new_distribution)
        new_distribution = remove_redundancies(new_distribution)
        #print("rr",new_distribution)
        new_distribution = [ pair[1] for pair in new_distribution]
        total = sum(new_distribution)
        new_distribution = [ n/total for n in new_distribution]
        new_distribution = bayesian_reconstruction.ProbabilityDistribution(N, p_func = new_distribution)
        # print("new_dist", new_distribution.p_func)
        # print("hellengier dist", hellinger_distance(current_distribution.p_func, new_distribution.p_func) )
        if hellinger_distance(current_distribution.p_func, new_distribution.p_func) < .00002:
            return new_distribution
        current_distribution = new_distribution
def jigsaw_example():
    N=14
    dist = bayesian_reconstruction.ProbabilityDistribution(N, bayesian_reconstruction.uniform_sample_simplex(N))
    random_to_compare = bayesian_reconstruction.ProbabilityDistribution(N, bayesian_reconstruction.uniform_sample_simplex(N))
    pmeas = simulate_partial_measurements(dist, generate_partial_measurements(range(N), 2), N,sample_size=100)
    fmeas = simulate_partial_measurements(dist, generate_partial_measurements(range(N), 1), N)
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
jigsaw_example()
    #TODO: Complete jigsaw reconstruction
#print(set_of_outcomes_to_integer({0},10), set_of_outcomes_to_integer({0,1},10))
