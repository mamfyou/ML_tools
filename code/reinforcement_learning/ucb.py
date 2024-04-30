import math
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('../../data_sets/Ads_CTR_Optimisation.csv')

round = 1000

ads = 10

sum_rewards = [0] * ads
numbers_of_selections = [0] * ads
selected_ads = []
total_reward = 0

for r in range(round):
    ad = 0
    max_upper_bound = 0
    for selected_ad in range(ads):
        if numbers_of_selections[selected_ad] > 0:
            avg_reward = sum_rewards[selected_ad] / numbers_of_selections[selected_ad]
            delta_i = math.sqrt(
                3 / 2 * math.log(r + 1) / numbers_of_selections[selected_ad]
            )
            upper_bound = avg_reward + delta_i
        else:
            upper_bound = 1e400

        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = selected_ad

    numbers_of_selections[ad] += 1
    reward = dataset.values[r, ad]
    sum_rewards[ad] += reward
    selected_ads.append(ad)
    total_reward += reward

plt.hist(selected_ads)
plt.xlabel('Ads')
plt.ylabel('upper bound')
plt.show()
