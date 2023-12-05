#The goal of this script it to take a list of partial outcomes and return the most likely distrribution that generated.
import math
import numpy as np
import mcint
import random

class ProbabilityDistribution:
    def __init__(self, N, p_func=None):
        if not isinstance(N, int) or N <= 0:
            raise ValueError("N must be a positive integer.")
        self.N = N
        self.p_func = p_func or self.default_p_func

    def default_p_func(self, k):
        if not isinstance(k, int) or k < 0 or k > self.N:
            raise ValueError(f"Invalid input k. It must be an integer between 0 and {self.N}.")
        return 1 / self.N

    def sum_p_func_over_event(self, event):
        if not isinstance(event, Event):
            raise ValueError("Input must be an instance of Event.")
        return sum(self.p_func(outcome) for outcome in event.outcomes)
    def sample(self, size=1):
        if not isinstance(size, int) or size <= 0:
            raise ValueError("Size must be a positive integer.")
        
        # Generate random samples based on probabilities defined by p_func
        samples = np.random.choice(range(self.N), size=size, p=[self.p_func(k) for k in range(self.N)])
        return samples

# Example usage:
N_value = 5
prob_dist_default = ProbabilityDistribution(N_value)
for i in range(N_value + 1):
    print(f"Default P({i}) = {prob_dist_default.p_func(i)}")

class Event:
    def __init__(self, outcomes, N, confidence=1.0):
        if not isinstance(outcomes, set) or not all(isinstance(outcome, int) for outcome in outcomes):
            raise ValueError("Outcomes must be a set of integers.")
        
        if not isinstance(N, int) or N <= 0:
            raise ValueError("N must be a positive integer.")
        
        if not all(0 <= outcome <= N-1 for outcome in outcomes):
            raise ValueError("Elements of outcomes must be less than or equal to N-1.")
        
        if not isinstance(confidence, (int, float)) or confidence < 0:
            raise ValueError("Confidence must be a non-negative float.")

        self.outcomes = outcomes
        self.N = N
        self.confidence = confidence

    def probability(self):
        return len(self.outcomes) / self.N

class MetaDistribution:
    def __init__(self, N, dist_func=None):
        if not isinstance(N, int) or N <= 0:
            raise ValueError("N must be a positive integer.")
        self.N = N
        self.dist_func = dist_func or self.default_dist_func

    def default_dist_func(self, prob_dist):
        if not isinstance(prob_dist, ProbabilityDistribution):
            raise ValueError("Input must be an instance of ProbabilityDistribution.")
        
        # Replace the following line based on your specific requirements
        return 1 / math.factorial(self.N)

    def bayesian_update(self, event):
        if not isinstance(event, Event):
            raise ValueError("Event must be an instance of Event.")
        
        likelihood = self.dist_func.sum_p_func_over_event(event)  # P(Event | Hypothesis)
        prior = self.dist_func  # P(Hypothesis)
        evidence = event.probability()  # P(Event)

        # Applying Bayes' rule: P(Hypothesis | Event) = P(Event | Hypothesis) * P(Hypothesis) / P(Event)
        posterior = likelihood * prior / evidence

        # Update the dist_func attribute with the new posterior distribution
        self.dist_func = lambda prob_dist: posterior

        return self

# Example usage for Bayesian update:
N_value_meta = 5
meta_dist = MetaDistribution(N_value_meta)

# Create an Event instance
outcomes_set_event = {0, 1, 2}
confidence_value_event = 0.8
event_instance = Event(outcomes=outcomes_set_event, N=N_value_meta, confidence=confidence_value_event)

# Perform Bayesian update
meta_dist.bayesian_update(event_instance)

# Test the updated dist_func
prior_prob_dist = ProbabilityDistribution(N_value_meta)
for i in range(N_value_meta + 1):
    print(f"Updated P({i}) = {meta_dist.dist_func(prior_prob_dist)}")

