#!/usr/bin/env python
# coding: utf-8

# # Q 2. Answer the following questions for above architecture:

# ## 1. What are the number of parameters in 1st convolutional layers ?

# **Solution:** Number of parameters in first conv layer: (3*5*5 + 1)*6 = 456

# ## 2. What are the number of parameters in pooling operation?

# **Solution:** <br>
# Filter = 2x2 ,strid = 2 <br>
# Number of parameters in first pooling operation: (1+1)*6 = 12

# ## 3. Which of the following operations contain most number of parameters? 
# ### (a) conv (b) pool (c) Fully connected layer (FC) (d) Activation Functions

# **Solution:** <br>
# Number of parameters in first conv layer: (3x5x5 + 1)x6 = 456, output = 6 @ 28 x 28 <br>
# Number of parameters in first activation layer: 6x28x28 = 1568, output = 6 @ 28 x 28 <br>
# Number of parameters in first pooling layer: (1+1)x6 = 12, output = 6 @ 14 x 14 <br>
# <br>
# Number of parameters in second conv layer: (6x5x5 + 1)x16 = 2416, output = 16 @ 10 x 10 <br>
# Number of parameters in second activation layer: 16x10x10 = 1600, output = 16 @ 10 x 10 <br>
# Number of parameters in second pooling layer: (1+1)x16 = 32, output = 16 @ 5 x 5 <br>
# <br>
# <br>
# Number of parameters in first fully connected conv layer: (16x5x5)x120 + 120 = 48120, output = 120<br>
# Number of parameters in first activation fully connected conv layer: 120x1 = 120, output =120 <br>
# <br>
# Number of parameters in second fully connected layer: 120x84 + 84 = 10164, output = 84<br>
# Number of parameters in second activation fully connected layer: 120x1 = 84, output = 84 <br>
# <br>
# Number of parameters in output layer: 84x10 + 10 = 850, output = 10<br>
# Number of parameters in output activation layer: 10x1 = 10, output = 10 <br>
# <br>
# 
# **conclusion:**<br>
# * conv = 456 + 2416 = 2872<br>
# * pool = 12 + 32 = 44<br>
# * Fully connected layer = 48120 + 10164 = 58284<br>
# * Activation Functions = 1568 + 1600 + 120 + 84 + 10 = 3298<br><br>
# 
# **Fully connected layer** has the most number of parameters

# 
# ## 4. Which operation consume most amount of memory?
# ### (a) initial convolution layers (b) fully connected layers at the end

# **Solution:**<br>
# Using the results form previous solution, <br>
# Memory usage for inital convolution layers = 456x28x28 + 2416x10x10 = **599104**<br>
# Memory usage for fully connected layers = 48120x120 + 10164x84 = **6628176** <br><br>
# **Conclusion:**<br> **fully connected layers** consumes most amount of memory

# In[ ]:




