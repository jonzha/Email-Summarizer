import random
from random import randint
import math
from operator import add
import numpy as np
class Individual:
    weights = []
    rankings = []
    def __init__(self, weights):
        self.weights = weights

    def to_string(self):
        return self.weights

    def calculate_score(self, parameters):
        return sum([a * b for a, b in zip(self.weights, parameters)])

    # Takes in text, which is a list of sentences
    def calculate_rankings(self, text):
        score_list = []
        for i in range(0, len(text)):
            curr_sent = text[i]
            score_list.append( self.calculate_score(text[i].get_parameters()))
        self.rankings = [i[0] for i in sorted(enumerate(score_list), key=lambda x: x[1])]


# creates an individual object
def make_individual():
    return Individual([random.uniform(-1, 1) for i in range(0, 4)])

# Creates a population of size count
def population(count):
    return [ make_individual() for x in xrange(count) ]

# Takes in an individual and calculates its fitness based off a target set of rankings
def fitness(individual, target):
    return euclidean(individual.rankings, target)

def pop_fitness(population, target):
    summed = reduce(add, (fitness(x, target) for x in population), 0)
    return summed / float(population)

def evolve(population, target, retain=0.2, random_select=0.05, mutate=0.01):

    # Generate the parents
    graded = [ (fitness(x, target), x) for x in population]
    graded = [ x[1] for x in sorted(graded)]
    retain_length = int(len(graded)*retain)
    parents = graded[:retain_length]

    # Add in other individuals
    for individual in graded[retain_length:]:
        if random_select > random.random():
            parents.append(individual)

    # Mutate
    for individual in parents:
        if mutate > random.random():
            parameter_to_mutate = randint(0, len(individual.weights)-1)
            individual.weights[parameter_to_mutate] = randint(-1, 1)

    # Generate children
    parents_len = len(parents)
    desired_len = len(population) - parents_len
    children = []
    while len(children) < desired_len:
        father = randint(0, parents_len-1)
        mother = randint(0, parents_len-1)
        if father != mother:
            father = parents[father]
            mother = parents[mother]
            half = len(father.weights) / 2
            child = Individual(father.weights[:half] + mother.weights[half:])
            children.append(child)
    parents.extend(children)
    return parents

# Takes in an two lists a and b of same length and returns their euclidean  distance
def euclidean(a, b):
    distance = 0
    for i in range(0, len(a)):
        distance += math.pow(a[i] - b[i], 2)
    return math.sqrt(distance)