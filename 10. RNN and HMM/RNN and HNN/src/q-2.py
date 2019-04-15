#!/usr/bin/env python
# coding: utf-8

# In[1]:


list_of_nucleotides = ["A", "C", "G", "T"]
print(list_of_nucleotides)


# In[2]:


dict_prob_emission = { "E" : {"A":0.25,"C":0.25,"G":0.25,"T":0.25}, "5":{"A":0.05, "C":0, "G":0.95, "T":0}, "I":{"A":0.4, "C":0.1, "G":0.1, "T":0.4}}
dict_prob_transition = { "^" : {"E":1},"E":{"E":0.9,"5":0.1},"5":{"I":1},"I":{"I":0.9,"$":0.1}}
sequence =   "CTTCATGTGAAAGCAGACGTAAGTCA"
state_path = "EEEEEEEEEEEEEEEEEE5IIIIIII$"
list_sequence = list(sequence)
list_state_path = list(state_path)
print(list_sequence)
print(list_state_path)


# In[3]:


prob = dict_prob_transition["^"][state_path[0]]
print(prob)
import math
from decimal import *
getcontext().prec = 7

for i in range(0, len(list_sequence)):
    s0 = state_path[i]
    s1 = state_path[i+1]
    n = list_sequence[i]


    e = dict_prob_emission[s0][n]
    print(" s0:",s0," n:",n," e:",e,"\n")
    t = dict_prob_transition[s0][s1]
    print(" s0:",s0," s1:",s1," t:",t,"\n")

    prob = Decimal(prob) * Decimal(e) * Decimal(t)
    print(prob)
print(math.log(prob))
 

# print(log(prob))


# In[ ]:





# In[ ]:





# In[ ]:




