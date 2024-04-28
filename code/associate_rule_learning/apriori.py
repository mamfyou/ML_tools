import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from apyori import apriori

dataset = pd.read_csv('../../data_sets/transaction3.csv', header=None)

order_items = []
for i in range(dataset.shape[0]):
    order_items.append([str(dataset.values[i, j]) for j in range(12)])

rules = apriori(transactions=order_items, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2, max_length=2)
results = list(rules)
print(results)


def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))


results_in_data_frame = pd.DataFrame(inspect(results),
                                     columns=['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])
print(results_in_data_frame.nlargest(10, 'Lift'))
