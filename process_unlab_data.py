import numpy as np
import os
import json
from tqdm import tqdm
import csv
import string

with open('data.json', 'r') as fi:
    [tweet_ids, tweet_id2topic]=json.load(fi)
fi.close()
with open('users.json') as d:
    users = json.load(d)
d.close()
dataset = []

for filename in tqdm(os.listdir('data')):
    with open('data\\'+filename, 'r', encoding='utf8') as f:
        tweet_data = json.load(f)
    f.close()
    for i in range(len(tweet_data)):
        try:
            if tweet_data[i]['id'] in tweet_ids:
                dataset.append(tweet_data[i])
        except KeyError:
            continue

for tweet in tqdm(dataset):
    hasAuthor = False
    for record in users:
        try:
            for account in record['accounts']:
                if tweet['screen_name'] == account['screen_name']:
                    if account['party'] == 'R':
                        tweet['author'] = 'republican'
                        hasAuthor = True
                    else:
                        tweet['author'] = 'democrat'
                        hasAuthor = True
        except IndexError:
            continue
    if hasAuthor == False:
        tweet['author'] = 'na'
        
        
dataset_fin = []
for i in range(len(dataset)):
    try:
        dataset_fin.append([dataset[i]['id'], tweet_id2topic[dataset[i]['id']], dataset[i]['text'], dataset[i]['author'], None])
    except KeyError:
        dataset_fin.append([dataset[i]['id'], tweet_id2topic[dataset[i]['id']], dataset[i]['text'], 'na', None])


for i in range(len(dataset_fin)):
    dataset_fin[i][2] = dataset_fin[i][2].replace('\n', ' ')
    dataset_fin[i][2] = dataset_fin[i][2].replace('\t', ' ')
    dataset_fin[i][2] = dataset_fin[i][2].encode('ascii', errors='ignore').decode()

file = open('train_extra.csv', 'a+', newline='', encoding='utf8')
with file:
    write = csv.writer(file)
    write.writerows(dataset_fin)




