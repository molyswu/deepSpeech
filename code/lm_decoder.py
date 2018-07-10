"""
Author: Lisa Scaria - lscaria@ford.com
"""
import copy
import heapq
import kenlm
import numpy as np
from lm import LM
import collections
import math
from collections import defaultdict
from datetime import datetime
from asr.align import calculate_wer
from Levenshtein import distance
import deepSpeech_input

ALPHABET = deepSpeech_input.ALPHABET
IX_TO_CHAR = {i: ch for (i, ch) in enumerate(ALPHABET)}
CHAR_TO_IX = {ch: i for (i, ch) in enumerate(ALPHABET)}
blank_idx = len(ALPHABET)


"""
Module for implementing external decoders in Python.

Note: This is simply for experimentation. Neither the PrefixDecoder
or the GreedyDecoder are designed for optimal results.
"""

class PrefixDecoder():
    def __init__(self, lm_model):
        self.lm_model = lm_model

    def lm_beam_decode(self, logits, beam_width=40, alpha=0.25, beta=0.25 , batch = 0):
        """
        Implementation based on https://arxiv.org/abs/1408.2873
        :param logits: num_time frames x num_characters numpy matrix of logits
        """
        num_timesteps = logits.shape[0]
        A_prev = [""]
        p_t_b = defaultdict(lambda: 1.0)
        p_tm1_b = defaultdict(lambda: 1.0)
        # Non-blank probabilities
        p_t_nb = defaultdict(float)
        p_tm1_nb = defaultdict(float)
        for char in ALPHABET:
            p_t_nb[char] = 0
            p_tm1_nb[char] = 0


        for t in range(num_timesteps):
            A_next = []
            A_dict = defaultdict()
            for l in A_prev:

                # Using "#" to denote blank
                for idx, c in enumerate(ALPHABET + "#"):
                    samechar = False
                    if c == "#":
                        # Should be 29 -- last idx in alphabet

                        p_t_b[l] = logits[t][batch][blank_idx] * (p_tm1_b[l] + p_tm1_nb[l])
                        A_next.append([p_t_b[l] + p_t_nb[l], l])
                        # if l in A_dict:
                        #     A_dict[l] = (p_t_b[l] + p_t_nb[l]) *A_dict[l]
                        # else:
                        #     A_dict[l] = p_t_b[l] + p_t_nb[l]

                    else:
                        l_plus = l + c
                        c_idx = CHAR_TO_IX[c]
                        if l.endswith(c):
                            samechar = True

                            p_t_nb[l_plus] = logits[t][batch][c_idx]*p_tm1_b[l]
                            p_t_nb[l] = logits[t][batch][c_idx]*p_tm1_b[l]
                            A_next.append([p_t_nb[l_plus] + p_t_b[l_plus], l])
                            # if l in A_dict:
                            #     A_dict[l_plus] = (p_t_nb[l_plus] + p_t_b[l_plus]) * A_dict[l]
                            # else:
                            #     A_dict[l_plus] = p_t_nb[l_plus] + p_t_b[l_plus]

                        # Space so query language model
                        elif c == " ":
                            p_t_nb[l_plus] = np.power(self.lm_model.score(l_plus.lower()), -10)**alpha * logits[t][batch][idx] * (p_tm1_b[l] + p_tm1_nb[l])
                            #p_t_nb[l_plus] = logits[t][1][c_idx] * (p_tm1_b[l] + p_tm1_nb[l])

                        else:
                            ##print("Char")
                            p_t_nb[l_plus] = logits[t][batch][c_idx] * (p_tm1_b[l] + p_tm1_nb[l])

                        if l_plus not in A_prev:
                            p_t_b[l_plus] = logits[t][batch][blank_idx] * (p_tm1_b[l] + p_tm1_nb[l])
                            if l == "":
                                p_t_nb[l_plus] = logits[t][batch][c_idx] #* p_tm1_nb[l]
                            else:
                                p_t_nb[l_plus] = logits[t][batch][c_idx] * p_tm1_nb[l]
                        if not samechar:
                            A_next.append([p_t_nb[l_plus] + p_t_b[l_plus], l_plus])
                            # if l_plus in A_dict:
                            #     A_dict[l_plus] = (p_t_nb[l_plus] + p_t_b[l_plus]) * A_dict[l_plus]
                            # else:

                            #     A_dict[l_plus] = p_t_nb[l_plus] + p_t_b[l_plus]

            # Add word count penalty
            for idx, an in enumerate(A_next):
                num_words = len(an[1].split())
                if num_words == 0:
                    penalty = 0.
                else:
                    penalty = num_words

                A_next[idx][0] *= (penalty ** beta)
            A_next = sorted(A_next, key=lambda p: p[0], reverse=True)

            # Take beam_width most probable sequences
            A_prev = [word[1] for word in A_next[:beam_width]]
            p_tm1_nb = copy.deepcopy(p_t_nb)
            p_tm1_b = copy.deepcopy(p_t_b)

        return A_prev[0]


    def decode_logits(self, logits, beam_width =40, alpha = 0.25, beta = 0.25):

        beam_label = []
        for i in range(logits.shape[1]):
            beam  = self.lm_beam_decode(logits, beam_width, alpha, beta,batch = i)
            beam_label.append((remove_repeats(beam)))

        return beam_label



class GreedyDecoder():
    def greedy_decode(self,logits, batch =0):

        blank_idx = len(ALPHABET)
        num_timesteps = logits.shape[0]
        s = ""
        for t in range(num_timesteps):
            ix = np.argmax(logits[t][batch])
            if(ix!=blank_idx):
                s = s + IX_TO_CHAR[ix]

        return s

    def decode_logits(self, logits):

        greedy_label = []
        for i in range(logits.shape[1]):
            greedy  = self.greedy_decode(logits, i)
            greedy_label.append(remove_repeats(greedy).upper())

        return greedy_label

def remove_repeats(output):
     words = output.split()

     outarray = []
     for i in range(len(words)):
        word = words[i]
        newword = []
        c=1
        newword.append(word[0])
        prev = word[0]
        while c < len(word)-1:
            if(word[c] != prev ):
                newword.append(word[c])
                prev = word[c]
            c = c+1
        if len(word)>2 and word[-1] != word[-2]:
            newword.append(word[-1])
        words[i] = ''.join(newword)
     return ' '.join(words)




if __name__ == "__main__":
    lm = LM("/home/lscaria/Random/deepSpeech/LanguageModels/wsj.klm")
    logits = np.load('out.txt.npy')
    with open("labels.txt") as f:
        labels = f.readlines()

    labels = [x.strip() for x in labels]
    print(logits.shape)

    greedyDecoder = GreedyDecoder()
    decoded = greedyDecoder.decode_logits(logits)
    print(decoded)
    # prefixDecoder = PrefixDecoder(lm)
    # decoded = prefixDecoder.decode_logits(logits)
