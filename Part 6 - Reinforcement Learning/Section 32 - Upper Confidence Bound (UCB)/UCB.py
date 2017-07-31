#Upper Confidence Bound RL

#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

#Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementing UCB
N = 10000
d = 10
ads_selected  = []
num_of_selections = [0] * d
sums_of_rewards = [0] * d
total_reward = 0
#N is the total number of rounds
#d is the number of ads

for n in range(0,N):
    max_upper_bound = 0
    ad = 0
    for i in range(0,d):
        if num_of_selections[i] > 0:
            average_reward = sums_of_rewards[i] / num_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1)/num_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    num_of_selections[ad] += 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] += reward
    total_reward += reward
    
#Visualizing the results
plt.hist(ads_selected)
plt.title('Histogram of ad selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()