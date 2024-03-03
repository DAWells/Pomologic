"""
View embeddings with umap
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import umap

details = pd.read_csv("data/interim/tidy_details.csv", index_col=0)
# Keep only common fruit
common_fruit = ['apples', 'quinces', 'strawberries', 'oranges', 'lemons', 'grapefruits', 'loquats', 'figs', 'persimmons', 'tangelos']
details = details[details.fruit.isin(common_fruit)]
details.fruit = pd.Categorical(details.fruit)

embeddings = pd.read_csv("data/interim/embeddings.csv", index_col=0, header=None)

df = pd.merge(details, embeddings, how="inner", left_index=True, right_index=True)

reducer = umap.UMAP()
proj = reducer.fit_transform(df.iloc[:, details.shape[1]:].values)

fig, ax = plt.subplots()
for fruit in df.fruit.cat.categories:
    fruit
    mask = df.fruit == fruit
    ax.scatter(
        proj[mask, 0],
        proj[mask, 1],
        label = fruit
    )
ax.legend()
plt.show()

####################
# Apple embeddings
####################
apples = df[df.fruit == "apples"]
apples.variety = pd.Categorical(apples.variety)
reducer = umap.UMAP()
proj = reducer.fit_transform(apples.iloc[:, details.shape[1]:].values)

fig, ax = plt.subplots()
ax.scatter(
    proj[:, 0],
    proj[:, 1],
    c = apples.variety.cat.codes,
    cmap = cm.plasma
)
ax.legend()
plt.show()


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
fruits = details.groupby("fruit").size().sort_values().index.tolist()

for i,fruit in enumerate(fruits):
    x = year_count[fruits[:i+1]].cumsum().apply(sum, axis=1)
    totals[fruit] = x

fig, ax = plt.subplots()
for i,fruit in enumerate(fruits):
    ax.plot(year_count.year, totals[fruit], label=fruit)
    ymax = totals[fruit]
    ymin = totals[fruits[i-1]]
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
import networkx as nx
G = nx.Graph()

details = details[:100]


edges = []
pom_nodes = []
for i,row in details.iterrows():
    edges.append((i, row.author))
    pom_nodes.append((i, {'fruit': row.fruit}))

author_nodes = details.author.unique()

G.add_nodes_from(pom_nodes)
G.add_nodes_from(author_nodes)

G.add_edges_from(edges)

nx.draw(G,node_color="fruit")
plt.show()

