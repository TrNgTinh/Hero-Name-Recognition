import requests, json, cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# download and save data file for LOL champions. The "champion.json" returns a list of champions with a brief summary.
url="http://ddragon.leagueoflegends.com/cdn/12.12.1/data/en_US/champion.json"
reqs = requests.get(url)
dictionary = reqs.json()

# serializing json (dict to str)
json_object = json.dumps(dictionary, indent = 2)

# filter all the LOL champion names
championNames = dictionary['data'].keys()

for name in championNames:
    # download all the LOL champion images(png)
    url="http://ddragon.leagueoflegends.com/cdn/12.12.1/img/champion/" + name + ".png"
    reqs = requests.get(url)
    path_root = "Champions//" 
    os.makedirs(path_root, exist_ok = True)
    with open( path_root + name + ".png", "wb") as file:  # write and binary mode
        file.write(reqs.content) 