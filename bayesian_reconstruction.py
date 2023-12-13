#The goal of this script it to take a list of partial outcomes and return the most likely distrribution that generated.
import math
import numpy as np
import mcint
import random
from scipy.optimize import minimize, LinearConstraint
epsilon =0.001

def simplex_volume(N):
    #returns the volume of a simplex with N outcomes
    #We use the formula here:https://www.youtube.com/watch?v=o9hXjCpVB10
    #Keep in mind that the simplex has dimension N-1 and side-length sqrt(2).
    return math.sqrt(N)/math.factorial(N-1)

def uniform_sample_simplex(N):
    #implements https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex
    #retuns a length N array, representing a probability distribution piked uniformly at random.
    random_numbers =np.concatenate( (np.random.uniform(size = N-1), np.array([0,1])), axis = None)
    random_numbers.sort()
    output = [random_numbers[i+1]-random_numbers[i] for i in range(len(random_numbers)-1)]
    return output

class ProbabilityDistribution:
    def __init__(self, N, p_func=None):
        if not isinstance(N, int) or N <= 0:
            raise ValueError("N must be a positive integer.")
        self.N = N #Number of possible outcomes
        self.p_func = p_func or self.default_p_func #A function that gives the probability of each outcome.


    def default_p_func(self, k):
        if not isinstance(k, int) or k < 0 or k > self.N:
            raise ValueError(f"Invalid input k. It must be an integer between 0 and {self.N}.")
        return 1 / self.N

    def sum_p_func_over_event(self, event): #Returns the probability of the event for this probability distribution.
        if not isinstance(event, Event):
            raise ValueError("Input must be an instance of Event.")
        if not event.N==self.N:
            raise ValueError("Event must have the same number of outcomes as distribution.")
        return sum(self.p_func(outcome) for outcome in event.outcomes)
    def sample(self, size=1):
        if not isinstance(size, int) or size <= 0:
            raise ValueError("Size must be a positive integer.")
        # Generate random samples based on probabilities defined by p_func
        samples = np.random.choice(range(self.N), size=size, p=self.p_func)
        return samples
    def marginal(self,outcomes):
        #expects outcomes to be a list of elements [0,1...N-1]
        outcomes = list(outcomes)
        outcomes.sort()
        marginal_dist = [ self.p_func[o] for o in outcomes]
        denom = sum(marginal_dist)
        if np.isclose(denom, 0):
            return ProbabilityDistribution(new_N, p_func = [ 1/new_N ]*new_N)
        marginal_dist = [m/denom for m in marginal_dist]
        new_N= len(outcomes)
        return ProbabilityDistribution(new_N, p_func = marginal_dist)

class Event: #Describes a partial measurement of outcomes.
    def __init__(self, outcomes, N, confidence=1.0):
        if not isinstance(outcomes, set) or not all(isinstance(outcome, int) for outcome in outcomes):
            raise ValueError("Outcomes must be a set of integers.")

        if not isinstance(N, int) or N <= 0:
            raise ValueError("N must be a positive integer.")

        if not all(0 <= outcome <= N-1 for outcome in outcomes):
            raise ValueError("Elements of outcomes must be less than or equal to N-1.")

        if not isinstance(confidence, (int, float)) or confidence < 0:
            raise ValueError("Confidence must be a non-negative float.")

        self.outcomes = outcomes #A set that describes the the event that some element of outcomes occured.
        self.N = N #The total number of outcomes.
        self.confidence = confidence #A number between 0 and 1. We assume that the outcome is drawn uniformly at randdom with probability (1-confidence) and drawn from an unknown distribution with probability confidence.

    def probability(self):#The probability of the event occuring, given that the distribution is uniformly random.
        return len(self.ou_value_metatcomes) / self.N

#Don't use any code from this file after this point because it's buggy

class MetaDistribution: #A class that a gives each ProbabilityDistribution a probability of occurring.
    def __init__(self, N, dist_func=None,center_of_mass=None, previous_centers_of_mass=[]):
        if not isinstance(N, int) or N <= 0:
            raise ValueError("N must be a positive integer.")
        self.N = N
        self.center_of_mass=None
        self.previous_centers_of_mass=previous_centers_of_mass
        self.dist_func = dist_func or self.default_dist_func() #dist_func is analogous to p_func. It is a distribuution on ProbabilityDistributions.
    def default_dist_func(self): #Default to the uniform distribution.
        return lambda x: 1/simplex_volume(self.N)

    def set_center_of_mass_axis(self,ax):#use montecarlo integration to calculate the center of mass of dist_func.
        #expects ax to be in [0...N-1]
        def integrand(x):
            return(x[ax]*self.dist_func(x))
        def sampler():
            while True:
                yield np.array(uniform_sample_simplex(self.N))
        result, error = mcint.integrate(integrand, sampler(), measure=simplex_volume(self.N), n=1000000)
        self.center_of_mass=result
        return(result)
    def set_center_of_mass(self):
        if not self.center_of_mass is None:
            raise ValueError("center of mass set twice")
        self.center_of_mass=[self.set_center_of_mass_axis(ax) for ax in range(self.N)]
        self.previous_centers_of_mass.append(self.center_of_mass)

    def bayesian_update(self, event): #something seems wrong. Maybe compounding numerical issues?
        if self.center_of_mass is None:
            self.set_center_of_mass()
        if not isinstance(event, Event):
            raise ValueError("Event must be an instance of Event.")
        def new_dist_func(prob_dist):

            likelihood = sum( {prob_dist[e] for e in event.outcomes})  # P(Event | Hypothesis)
            if np.isclose(likelihood, 0):
                return 0
            log_to_return = math.log2(likelihood)*len(self.previous_centers_of_mass)- math.log2(sum({self.center_of_mass[outcome] for outcome in event.outcomes })) - math.log2(simplex_volume(self.N))
            return(2**log_to_return)
        return(new_dist_func)

    def bayesian_updates(self, events):#expects events to be a set of events
        def new_dist_func(prob_dist):
            try:
                log_likelihood = sum( {math.log2(sum({prob_dist[o] for o in e.outcomes})) for e in events})
            except:
                print("aaccepted")
                return(0)
            log_evidence = sum({ math.log2(1/self.N) for e in events})
            log_prior = math.log2(1/simplex_volume(self.N))
            return(2**(log_likelihood-log_evidence +log_prior))
        return(new_dist_func)

    def most_likely_distribution(self):
        initial_guess = uniform_sample_simplex(self.N)
        constraint_matrix = np.ones(self.N)
        constraint_rhs = np.array([1])
        linear_constraint = LinearConstraint(constraint_matrix, lb=np.array([1]), ub=np.array([1]))
        variable_bounds = [(0,1)]*self.N
        result = minimize(lambda x: -self.dist_func(x), initial_guess, method='SLSQP',constraints=(linear_constraint), bounds=variable_bounds)
        return(result.x)

def full_outcomes_test():
    N=3
    x = uniform_sample_simplex(N)
    print("x", x)
    prob_dist = ProbabilityDistribution(N,p_func=x)
    samples = prob_dist.sample(size = 60)
    new_meta_dist = MetaDistribution(N)
    # for s in samples:
    #     print("sample", s)
    #     event = Event({int(s)},N)
    #     new_meta_dist=MetaDistribution(N, dist_func=new_meta_dist.bayesian_update(event), previous_centers_of_mass = new_meta_dist.previous_centers_of_mass)
    #     print(new_meta_dist.previous_centers_of_mass)
    #
    # print("actual value", x)
    # print("Bayesian guess", new_meta_dist.most_likely_distribution())
    samples = {Event({int(s)},N) for s in samples}
    new_meta_dist2 = MetaDistribution(N, dist_func = new_meta_dist.bayesian_updates(samples) )
    print(new_meta_dist2.most_likely_distribution())
    print("averages", [len([s.outcomes for s in samples if list(s.outcomes)[0]==i])/len(samples) for i in range(N)])
#full_outcomes_test()
#
# # Create an Event instance
# outcomes_set_event = {0, 1, 2}
# confidence_value_event = 0.8
# event_instance = Event(outcomes=outcomes_set_event, N=N_value_meta, confidence=confidence_value_event)
#
# # Perform Bayesian update
# meta_dist.bayesian_update(event_instance)
#
# # Test the updated dist_func
# prior_prob_dist = ProbabilityDistribution(N_value_meta)
# for i in range(N_value_meta + 1):
#     print(f"Updated P({i}) = {meta_dist.dist_func(prior_prob_dist)}")
