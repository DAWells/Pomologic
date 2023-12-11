# Pomologic

Download all water colours from the pomological USDA set
and create an infographic. Summarise the number of different fruits,
how many works each artist made and what they specialised in, finally
use umap/tsne to visualise embeddings of the images to see clustering.

Web scrape all of the pomological water colours from:
- https://search.nal.usda.gov/discovery/collectionDiscovery?vid=01NAL_INST:MAIN&collectionId=81279629860007426
or
- https://commons.wikimedia.org/wiki/Category:USDA_Pomological_Watercolors

## Download data
Download details about images and their src url from wikimedia.
Then download the images. Includes checks to avoid redownloading
details and images.

## Tidy details
Extract the date, author, fruit, and variety from details.

## Embed
Generate embeddings for images following
https://www.fuzzylabs.ai/blog-post/hugging-face-in-space

## Plots
Make plots