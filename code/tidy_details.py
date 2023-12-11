
"""Tidy details"""

import pandas as pd
import re

def get_year(date):
    date = str(date)
    pattern = re.compile("\d{4}")
    match = re.search(pattern, date)
    if match:
        match = match.group()
    return match

def get_author(author):
    author = str(author)
    pattern = re.compile(":  .*,")
    match = re.search(pattern, author)
    if match:
        match = match.group()[3:-1]
    return match

def get_description(description):
    description = str(description)
    pattern = re.compile("Image of the .+ \(scientific")
    match = re.search(pattern, description)
    if match:
        match = match.group()[13:-12]
    return match

def get_variety(description):
    description = str(description)
    pattern = re.compile("Image of the .+ variety")
    match = re.search(pattern, description)
    if match:
        match = match.group()[13:-8]
    return match

def get_fruit(description):
    description = str(description)
    pattern = re.compile("variety of .+ \(scientific")
    match = re.search(pattern, description)
    if match:
        match = match.group()[11:-12]
    return match


details = pd.read_csv("data/external/details.csv", names=["pom_id", "description", "author", "date", "image_src"])

details.date = details.date.apply(get_year)
details.author = details.author.apply(get_author)
details['fruit'] = details.description.apply(get_fruit)
details['variety'] = details.description.apply(get_variety)
details.description = details.description.apply(get_description)

details = details[['pom_id', 'date', 'author','fruit', 'variety','description']]
details.to_csv("data/interim/tidy_details.csv", index=False)