"""
View embeddings with umap
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import umap

details = pd.read_csv("data/interim/tidy_details.csv", index_col=0)
details.fruit = pd.Categorical(details.fruit)
embeddings = pd.read_csv("data/interim/embeddings.csv", index_col=0, header=None)

df = pd.merge(details, embeddings, how="right", left_index=True, right_index=True)

reducer = umap.UMAP()
proj = reducer.fit_transform(df.iloc[:, details.shape[1]:].values)

fig, ax = plt.subplots(2)
ax[0].scatter(
    proj[:,0],
    proj[:,1],
)

ax[1].scatter(
    proj[:,0],
    proj[:,1],
    c=df.fruit.cat.codes,
    cmap=cm.tab10
)
plt.show()

cm.tab10.get_bad()

####################
# What was painted when
####################

years = np.arange(
    int(details.date.min()),
    int(details.date.max())+1
)

year_count = []
for y in years:
    mask = details.date == y
    yc = details[mask].groupby("fruit", observed=False).size()
    yc = pd.DataFrame(yc).T
    yc['year'] = y
    year_count.append(yc)

year_count = pd.concat(year_count)

totals = {}
fruit = year_count.columns[:-1]
for i,f in enumerate(fruit):
    x = year_count[fruit[:i+1]].cumsum().apply(sum, axis=1)
    totals[f] = x

fig, ax = plt.subplots()
for i,f in enumerate(fruit):
    ax.plot(year_count.year, totals[f], label=f)
    ymax = totals[f]
    ymin = totals[fruit[i-1]]
    if i==0:
        ymin = 0
    ax.fill_between(
        year_count.year,
        ymax,
        ymin,
        alpha = 0.2
    )
ax.legend()
plt.show()



# Who painted what
# https://python-graph-gallery.com/500-network-chart-with-edge-bundling/