#!/usr/bin/env python
import os
import kenlm

"""
Author: Lisa Scaria
Modified from Kenlm example.py and Peter Nrovig SpellChecker
https://norvig.com/spell-correct.html
"""

class LM():

    def __init__(self, lm_path):
        self.model = kenlm.Model(lm_path)


#Check that total full score = direct score
    def score(self, s):
        return sum(prob for prob, _, _ in self.model.full_scores(s))

    def printNgram(self, sentence):
    	# Show scores and n-gram matches
    	words = ['<s>'] + sentence.split() + ['</s>']
    	for i, (prob, length, oov) in enumerate(self.model.full_scores(sentence, bos = False, eos = False)):
    	    print('{0} {1}: {2}'.format(prob, length, ' '.join(words[i+2-length:i+2])))
    	    if oov:
    	        print('\t"{0}" is an OOV'.format(words[i+1]))

    	# Find out-of-vocabulary words
    	for w in words:
    	    if not w in self.model:
    	        print('"{0}" is an OOV'.format(w))


    def edits1(self, word):
        "All edits that are one edit away from `word`."
        letters    = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ '
        splits     = [(word[:i], word[i:])    for i in range(0,len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        #print(set(deletes + transposes + replaces + inserts))
        return set(deletes + transposes + replaces + inserts)


    def edits2(self, word ,ed2):
        "All edits that are two edits away from `word`."
        #return (e2 for e1 in edits1(word) for e2 in edits1(e1))
        return (e2 for e1 in ed2 for e2 in self.edits1(e1))

    def edits3(self, edits2):
        "All edits that are two edits away from `word`."
        return (e3 for e3 in self.edits1(edits2))



    def known(self, words, size=0):
        "The subset of `words` that appear in the dictionary of WORDS."
        return set(w for w in words if self.model.__contains__(w) )

    def candidates(self, word):
        "Generate possible spelling corrections for word."
        ed1 = self.edits1(word)
        ed2 = self.edits2(word,ed1)
        return (self.known([word]) or self.known(ed1) or self.known(ed2)  or set([word]))
        #return (known([word]) or known(edits1(word))  or set([word]))


    def checkSentence(self, sentence):

    	s = sentence
    	words = sentence.split()
    	wordsdict = s.split()
    	for i in range(0,len(words)):
    		results = self.candidates(words[i])
    		initscore = -100
    		for result in results:
    			wordsdict[i] = result
    			newsentence = " ".join(wordsdict)
    			score = self.model.score(newsentence)
    			if(score>=initscore):
    				initscore = score
    				words[i] = result

    		wordsdict[i] = words[i]
    	return " ".join(words)



    def tryCombine(self, words):
        words = sentence.split()
        out = []

        for i in range(len(words)-1):
            base = self.model.score(words)


if __name__ == "__main__":
    lm = LM("../LM/wsj.klm")
    state = kenlm.State()
    state2 = kenlm.State()
    #Use <s> as context.  If you don't want <s>, use model.NullContextWrite(state).
    lm.model.BeginSentenceWrite(state)
    accum = 0.0
    accum += lm.model.BaseScore(state, "a", state2)
    accum += lm.model.BaseScore(state2, "sentence", state)
    #print accum
