"""
View nonapple embeddings with umap
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import umap

##########
# Embedbings for non-apple fruits
##########
naf_details = pd.read_csv("data/interim/tidy_nonapple_details.csv", index_col=0)
naf_details.fruit = pd.Categorical(naf_details.fruit)
naf_embeddings = pd.read_csv("data/interim/nonapple_embeddings.csv", index_col=0, header=None)

naf_df = pd.merge(naf_details, naf_embeddings, how="right", left_index=True, right_index=True)
# Filter the black and white illustrations
naf_df = naf_df['POM00000089':]
reducer = umap.UMAP()
naf_proj = reducer.fit_transform(naf_df.iloc[:, naf_details.shape[1]:].values)

# Colour scheme
cs = {
    'figs': "#6f5438",
    'grapefruits': "#e68c7c",
    'lemons': "#f5c04a",
    'loquats': "#c3c377",
    'oranges': "#fd8f24",
    'persimmons': "#4f5157",
    'quinces':"#919c4c",
    'strawberries': "#c03728",
    'tangelos':"#828585",
}

fig, ax = plt.subplots()
for fruit in naf_df.fruit.cat.categories:
    fruit
    mask = naf_df.fruit == fruit
    ax.scatter(
        naf_proj[mask, 0],
        naf_proj[mask, 1],
        label = fruit,
        c = cs[fruit]
    )
ax.legend()
plt.show()

# What are those weird strawberries?
mask = naf_df.fruit == "strawberries"
i = np.argmax(naf_proj[mask, 1])
naf_df[mask].iloc[i]

####################
# What was painted when
####################

years = np.arange(
    int(naf_details.date.min()),
    int(naf_details.date.max())+1
)

year_count = []
for y in years:
    mask = naf_details.date == y
    yc = naf_details[mask].groupby("fruit", observed=False).size()
    yc = pd.DataFrame(yc).T
    yc['year'] = y
    year_count.append(yc)

year_count = pd.concat(year_count)

totals = {}
fruits = naf_details.groupby("fruit").size().sort_values().index.tolist()

for i,fruit in enumerate(fruits):
    x = year_count[fruits[:i+1]].cumsum().apply(sum, axis=1)
    totals[fruit] = x

fig, ax = plt.subplots()
for i,fruit in enumerate(fruits):
    ax.plot(year_count.year, totals[fruit], label=fruit, c=cs[fruit])
    ymax = totals[fruit]
    ymin = totals[fruits[i-1]]
    if i==0:
        ymin = 0
    ax.fill_between(
        year_count.year,
        ymax,
        ymin,
        alpha = 0.2,
        color=cs[fruit]
    )
ax.legend()
plt.show()
