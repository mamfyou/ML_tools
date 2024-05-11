import random
import math
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('../../data_sets/Ads_CTR_Optimisation.csv')

round = 600
ads = 10

selected_ads = []
number_of_rewards_0 = [0] * ads
number_of_rewards_1 = [0] * ads
sum_of_rewards = 0

for n in range(round):
    selected_ad = 0
    max_random = 0
    for ad in range(ads):
        value = random.betavariate(number_of_rewards_1[ad] + 1, number_of_rewards_0[ad] + 1)
        if value > max_random:
            max_random = value
            selected_ad = ad

    selected_ads.append(selected_ad)
    reward = dataset.values[n, selected_ad]
    if reward == 1:
        number_of_rewards_1[selected_ad] = number_of_rewards_1[selected_ad] + 1
    else:
        number_of_rewards_0[selected_ad] = number_of_rewards_0[selected_ad] + 1

    sum_of_rewards += reward

plt.hist(selected_ads)
plt.title('Thompson Distribution')
plt.xlabel('Ads')
plt.ylabel('Frequency')
plt.show()
