# -*- coding: utf-8 -*-
from __future__ import division
import re
import nltk
from nltk import tokenize
from nltk.corpus import stopwords
import numpy

# Text is an object that represents a piece of text to be sumamrized. It contains a list of Simple_Sentences as well as
# a sentences_dic which holds the intersection score for each sentences.
class Text:
    sentences = []
    sentences_dic = []

    def __init__(self, text, title):
        self.text = text
        self.sentences = split_content_to_sentences(text)
        self.sentences_dic = createMatrix(text)

        # For relative length
        avg_length = 0
        for i in xrange(len(self.sentences)):
            avg_length += len(self.sentences[i])

        avg_length = float(avg_length) / len(self.sentences)

        for i in xrange(len(self.sentences)):
            self.sentences[i] = Simple_Sentence(self.sentences[i], self.sentences_dic[self.sentences[i]], i, avg_length,
                                                title)


# Simple_Sentence is an object that represents a sentence
class Simple_Sentence:
    # Text is the text, intersection is the intersection score from the sentences_dic and index is the index in which
    # the sentence appears in the original text
    def __init__(self, text, intersection, index, avg_length, title):
        self.text = text
        self.intersection = intersection
        self.index = index
        self.words = self.text.split(" ")
        self.num_words = len(self.words)
        self.len = len(self.text)
        self.relative_length = self.len / avg_length

        # Assumption: any words that start with capital letters are 'named entities' and are thus important
        # named_entities is proportion of words that start with capital letter
        named_entities = 0
        word_frequencies = {}
        title_words = title.split(" ")
        number_count = 0
        title_similarity = 0

        for word in self.words:
            if len(word) == 0:
                continue
            if word[0].isupper():
                named_entities += 1

            if word in word_frequencies:
                word_frequencies[word] += 1
            else:
                word_frequencies[0] = 0

            for title_word in title_words:
                if word == title_word:
                    title_similarity += 1

            if any(char.isdigit() for char in word):
                number_count += 1

        avg_word_frequency = 0
        for item in word_frequencies:
            avg_word_frequency += word_frequencies[item]

        self.average_tf = float(avg_word_frequency) / len(word_frequencies)
        self.named_entities = float(named_entities)
        self.title_similarity = float(title_similarity) / len(title_words)
        self.number_count = float(number_count) / self.num_words

    # Returns a list of parameters
    def get_parameters(self):
        return numpy.array([self.len, self.num_words, self.intersection, self.index, self.relative_length,
                self.named_entities, self.average_tf, self.title_similarity, self.number_count])
        # return [self.len, self.num_words, self.intersection, self.index, self.relative_length,
        #         self.named_entities, self.average_tf, self.title_similarity, self.number_count]

    def get_length(self):
        return self.len


def sentences_intersection(sent1, sent2):
    # split the sentence into words/tokens

    s1 = set(sent1.split(" "))

    s2 = set(sent2.split(" "))

    # If there is not intersection, just return 0

    if (len(s1) + len(s2)) == 0:
        return 0

    # We normalize the result by the average number of words
    return len(s1.intersection(s2)) / ((len(s1) + len(s2)) / 2)

    # Naive method for splitting a text into sentences


def split_content_to_sentences(content):
    return tokenize.sent_tokenize(content)
    #
    # content = content.replace("\n", ". ")
    # return content.split(". ")


# Naive method for splitting a text into paragraphs
def split_content_to_paragraphs(content):
    return content.split("\n\n")
    # Format a sentence - remove all non-alphbetic chars from the sentence
    #  We'll use the formatted sentence as a key in our sentences dictionary


def format_sentence(self, sentence):
    sentence = re.sub(r'\W+', '', sentence)
    return sentence


def createMatrix(text):
    sentences = split_content_to_sentences(text)
    n = len(sentences)
    values = [[0 for x in xrange(n)] for x in xrange(n)]
    for i in range(0, n):
        for j in range(0, n):
            values[i][j] = sentences_intersection(sentences[i], sentences[j])

    sentences_dic = {}
    for i in range(0, n):
        sum = 0
        for j in range(0, n):
            if i == j:
                continue
            sum += values[i][j]
        sentences_dic[sentences[i]] = sum
    return sentences_dic


def getBestSentences(originalText, text, num_sentences):
    original = split_content_to_sentences(originalText)
    sentences = split_content_to_sentences(text)
    sentences_dic = createMatrix(text)
    finalSummary = ""
    for i in range(0, num_sentences):
        curr_best = ""
        index = -1
        best_score = -1
        for j in range(0, len(sentences)):
            s = sentences[j]
            sOriginal = original[j]
            if sentences_dic[s] > best_score:
                curr_best = sOriginal
                best_score = sentences_dic[s]
                index = j
        del sentences[index]
        del original[index]
        finalSummary += " " + curr_best
    return finalSummary


def normalize(text):
    words = text.split(" ")
    for i, item in enumerate(words):
        words[i] = porter.stem(words[i])
    return ' '.join(words)


def removeStops(text):
    words = text.split(" ")
    cachedStopWords = stopwords.words("english")
    return ' '.join([word for word in text.split() if word not in cachedStopWords])


wnl = nltk.WordNetLemmatizer()
porter = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()

sent1 = "hello there my name is Jonathan. I'm having a running women in tables is are were verbs be was will has had"
sent2 = "hello there my name is Huigol"
text = "Princeton students take time off for a variety of reasons. Family, sports, health, religion, bridge year, or even to travel the world. We welcome all! Come enjoy a meal with a group who shares your experience and learn more! If you are planning to attend and/or would like to be added to our listserv, we kindly ask you to fill out this short form."
text = "Are you passionate about Asian and/or Pacific culture? Do you like organizing events? Do you like hanging out with cool people?? If the answer is yes to any of these questions, then you should apply to the Asian Pacific American Heritage Month (APAHM) 2017 Committee!! The APAHM Committee organizes a series of events celebrating Asian and Pacific American culture/identity throughout the month of April 2017. Our past events have included a talk by Nina Davuluri (Miss America 2014), and a conversation Amy Chua (Tiger Mom fame)."
text = "In all the experiments the training set consisted of 100 documents with their respective ideal automatic summaries. Experiments were carried out with two different kinds of test set. More precisely, in one experiment the test set consisted of 100 documents with their respective ideal automatic summaries, and in another experiment the test set consisted of 30 documents with their ideal manual summaries. In all experiments the training set and the test set were, of course, disjoint sets of documents, since the goal is to measure the predictive accuracy (generalisation ability) in the test set, containing only examples unseen during training. In order to evaluate how effective Genetic Algorithm (GA)-based attribute selection is in improving the predictive accuracy of ClassSumm, two kinds of GAs for attribute selection have been used – both of them following the wrapper approach. The first one was the Multi-Objective GA (MOGA) discussed in Section 2. MOGA was used to select attributes for J4.8, a well-known decision-tree induction algorithm [21]. Recall that MOGA performs a multi-objective optimisation (in the Pareto sense) of both J4.8’s error rate and the decision tree size. The results of training J4.8 with the attributes selected by the MOGA were compared with the results of training J4.8 with all original attributes, as a control experiment."
text = "In this day and age, everyone is barraged with a plethora of emails. Princeton students receive hundreds of emails each week. However, everyone is so busy that oftentimes many of these emails go unread and unnoticed. In fact, one of the authors, Zhang has over 10,000 unread emails. In order to tackle this problem, we decided to create a browser extension that will help summarize long emails, allowing users to quickly digest a large quantity of emails in a short amount of time. If successful, this would enable millions of people to not only keep their inboxes uncluttered, but also save them a good chunk of time that they would have spent on reading long emails. The end goal of this project is to have a Chrome extension that would hook into Gmail and summarize the user’s emails. There has been a fair amount of work done in the realm of text summary. There are two main forms of text summary: abstraction and extraction. Much work has been done with extraction, which simply lifts sentences from the given text, while not much work has been done with abstraction, which relies on natural language generation to generate a concise summary. One approach to extraction we found to be fascinating was through the use of a genetic algorithm (https://www.cs.kent.ac.uk/people/staff/aaf/pub_papers.dir/IBERAMIA-2004-Silla-Pappa.pdf). The way the algorithm works is it identifies certain features of each sentence such as the number of key words in it, or its position in the text and ranks each sentence based on the features. It then selects the highest ranking sentences to include in the summary. The genetic algorithm comes into play by optimizing the values of each feature. This is an algorithm we hope to explore in our work as well. "
text = text.replace("\xc2\xa0", " ")
text = text.decode('utf-8')
# Preprocessing stuff
# text = normalize(text.decode('utf-8'))
# [wnl.lemmatize(t) for t in sent1]
# words = text.split(" ")
# cachedStopWords = stopwords.words("english")
# text = ' '.join([word for word in text.split() if word not in cachedStopWords])
# print text
# for word in words:
#      word = wnl.lemmatize(word)
#      word = lancaster.stem(word)
#      word = porter.stem(word)
#      print "\n"
# text = ' '.join(words)
#
# textModified = normalize(text)
# # textModified = removeStops(text)
# textModified = textModified.lower();
# print textModified
# result = getBestSentences(text, textModified, 2)
#
#
# print result