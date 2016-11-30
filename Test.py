# -*- coding: utf-8 -*-

import Genetic as g
import Summary as s
text = "In this day and age, everyone is barraged with a plethora of emails. Princeton students receive hundreds of emails each week. However, everyone is so busy that oftentimes many of these emails go unread and unnoticed. In fact, one of the authors, Zhang has over 10,000 unread emails. In order to tackle this problem, we decided to create a browser extension that will help summarize long emails, allowing users to quickly digest a large quantity of emails in a short amount of time. If successful, this would enable millions of people to not only keep their inboxes uncluttered, but also save them a good chunk of time that they would have spent on reading long emails. The end goal of this project is to have a Chrome extension that would hook into Gmail and summarize the user’s emails. There has been a fair amount of work done in the realm of text summary. There are two main forms of text summary: abstraction and extraction. Much work has been done with extraction, which simply lifts sentences from the given text, while not much work has been done with abstraction, which relies on natural language generation to generate a concise summary. One approach to extraction we found to be fascinating was through the use of a genetic algorithm (https://www.cs.kent.ac.uk/people/staff/aaf/pub_papers.dir/IBERAMIA-2004-Silla-Pappa.pdf). The way the algorithm works is it identifies certain features of each sentence such as the number of key words in it, or its position in the text and ranks each sentence based on the features. It then selects the highest ranking sentences to include in the summary. The genetic algorithm comes into play by optimizing the values of each feature. This is an algorithm we hope to explore in our work as well. "
text = text.replace("\xc2\xa0", " ")
text = text.decode('utf-8')
text = s.Text(text)
sentences = text.sentences

# sentences = [s.Simple_Sentence("hello there"), s.Simple_Sentence("am"), s.Simple_Sentence("A big boy now")]
target = [5, 12, 6, 14, 0, 2, 1, 3, 4, 7, 8, 9, 10, 11, 13]

pop = g.population(200)
fitness_history = [g.pop_fitness(pop, target),]
best = 1000
for i in xrange(1000):
    #

    #     print j.weights
    #     print j.rankings
    #     print g.fitness(j, target)
    #     print "----"
    pop = g.evolve(pop, target)
    for j in pop:
        j.calculate_rankings(sentences)
    fitness = g.pop_fitness(pop,target)
    fitness_history.append(fitness)
    if fitness < best:
        best = fitness
        print pop[20].weights
        print pop[20].rankings
        print g.fitness(pop[20], target)
        print "---"
    print i
for datum in fitness_history:
   print datum

