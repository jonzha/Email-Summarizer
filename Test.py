import Genetic as g
import Summary as s
sentences = [s.Simple_Sentence("hello there"), s.Simple_Sentence("am"), s.Simple_Sentence("A big boy now")]
target = [1, 0, 2]
pop = g.population(5)
for i in pop:
    i.calc_rankings(sentences)
    print i.weights
    print i.rankings
    print g.fitness(i, target)
    print "----"

