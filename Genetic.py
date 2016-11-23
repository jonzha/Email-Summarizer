
import random
import math
from operator import add

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
    def calc_rankings(self, text):
        score_list = []
        for i in range(0, len(text)):
            curr_sent = text[i]
            score_list.append( self.calculate_score(text[i].get_parameters()))
        self.rankings = [i[0] for i in sorted(enumerate(score_list), key=lambda x: x[1])]


# creates an individual object
def make_individual():
    return Individual([random.uniform(-1,1)])

# Creates a population of size count
def population(count):
    return [ make_individual() for x in xrange(count) ]

# Takes in an individual and calculates its fitness based off a target set of rankings
def fitness(individual, target):
    return euclidean(individual.rankings, target)

def pop_fitness (population, target):
    summed = reduce(add, (fitness(x, target) for x in population), 0)
    return summed / (len(population) * 1.0)

# Takes in an two lists a and b of same length and returns their euclidean  distance
def euclidean(a, b):
    distance = 0
    for i in range(0, len(a)):
        distance += math.pow(a[i] - b[i], 2)
    return math.sqrt(distance)

