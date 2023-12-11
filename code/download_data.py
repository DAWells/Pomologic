"""
Download pomological watercolour dataset
"""
import os
import requests
from bs4 import BeautifulSoup as bs
import logging
import pandas as pd
import time

logging.basicConfig(level=logging.INFO)

# Saving interval
saving_interval = 10

def parse_details(soup, pom_id):
    description = soup.find_all("div", {"class": "description"})[0].text
    try:
        author = soup.find_all("div", {"class": "description"})[1].text
    except IndexError:
        author = ""
    try:
        date = soup.find("td", {"id": "fileinfotpl_date"}).find_next_sibling().text
    except AttributeError:
        date=""
    image_src = soup.find("table", {"class": "wikitable filehistory"}).find("img")['src']
    details = (pom_id, description, author, date, image_src)
    return details

# Load existing details to not download again
try:
    existing_details = pd.read_csv("data/external/details.csv", names=["pom_id", "description", "author", "date", "image_src"])
except FileNotFoundError:
    existing_details = pd.DataFrame(columns=["pom_id", "description", "author", "date", "image_src"])

# Download details
details_list = []
for i in range(50):
    pom_id = "POM" + str(i).zfill(8)
    logging.info(pom_id)
    url = f"https://commons.wikimedia.org/wiki/File:Pomological_Watercolor_{pom_id}.jpg"
    if pom_id in existing_details.pom_id.values:
        logging.info(f"Already got {pom_id}")
        continue
    try:
        page = requests.get(url)
        page.raise_for_status()
        soup = bs(page.content, "html.parser")
        details = parse_details(soup, pom_id)
        details_list.append(details)
    except requests.exceptions.HTTPError:
        logging.info(f"Error with {pom_id}")
    # Periodic saving
    if len(details_list) % saving_interval == 0:
        partial_details = pd.DataFrame(details_list, columns=["pom_id", "description", "author", "date", "image_src"])
        partial_details.to_csv("data/external/details.csv", header=False, index=False, mode="a")
        details_list = []
        logging.info("Partial details saved and reset")
    time.sleep(0.2)
else:
    partial_details = pd.DataFrame(details_list, columns=["pom_id", "description", "author", "date", "image_src"])
    partial_details.to_csv("data/external/details.csv", header=False, index=False, mode="a")


# Download images
details = pd.read_csv("data/external/details.csv", names=["pom_id", "description", "author", "date", "image_src"])

imgs = os.listdir("data/external/thumbnails")
for i,row in details.iterrows():
    pom_id = row.pom_id
    logging.info(pom_id)
    if f"{pom_id}.jpg" in imgs:
        logging.info(f"Already got {pom_id}")
        continue
    image_url = row.image_src
    img_data = requests.get(image_url).content
    with open(f'data/external/thumbnails/{pom_id}.jpg', 'wb') as handler:
        handler.write(img_data)
    time.sleep(0.2)

