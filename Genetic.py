import random
from random import randint
import math
from operator import add
import numpy

number_of_parameters = 9

class Individual:

    def __init__(self, weights):
        self.weights = weights
        self.rankings = []

    def to_string(self):
        return self.weights

    def get_rankings(self):
        return self.rankings

    def calculate_score(self, parameters):
        return numpy.dot(self.weights, parameters)
        # return sum([a * b for a, b in zip(self.weights, parameters)])

    # Takes in texts, which is a list of list of sentences
    def calculate_rankings(self, texts):
        self.rankings = []
        for text in texts:
            score_list = []
            for i in range(0, len(text)):
                curr_sent = text[i]
                score_list.append( self.calculate_score(text[i].get_parameters()))
            self.rankings.append([i[0] for i in sorted(enumerate(score_list), key=lambda x: x[1])])

# creates an individual object
def make_individual():
    # return Individual([random.uniform(-1, 1) for i in range(0, number_of_parameters)])
    return Individual(numpy.random.uniform(-1, 1, [number_of_parameters]))

# Creates a population of size count
def population(count):
    return [ make_individual() for x in xrange(count) ]

# Takes in an individual and calculates its fitness based off average of multiple target sets of rankings
def fitness(individual, target):
    curr_sum = 0
    for i in range(0, len(target)):
        curr_sum += euclidean(individual.get_rankings()[i], target[i])
    return curr_sum / len(target)

def pop_fitness(population, target):
    summed = reduce(add, (fitness(x, target) for x in population), 0)
    return float(summed) / len(population)

def evolve(population, target, retain=0.1, random_select=0.05, mutate=0.01):

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
            individual.weights[parameter_to_mutate] = numpy.random.uniform(-1, 1)

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
            child = Individual(numpy.append(father.weights[:half], mother.weights[half:]))
            # child = Individual(father.weights[:half] + mother.weights[half:])
            children.append(child)
    parents.extend(children)
    return parents

# Takes in an two lists a and b of same length and returns their euclidean  distance
def euclidean(a, b):
    # distance = 0
    # for i in range(0, len(a)):
    #     distance += math.pow(a[i] - b[i], 2)
    # return math.sqrt(distance)
    return numpy.linalg.norm(a-b)