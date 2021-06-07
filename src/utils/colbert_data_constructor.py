import json, os, re, requests
from es_functions import *
from tqdm import tqdm
import gdown
import argparse

"""
Generate FZxMedQA Dataset

This script will generate the FZxMedQADataset using dataset files with questions linked to FindZebra Corpus (using CUIs)
A running instance of ElasticSearch 7.13 must be running on localhost:9200

Run 'docker compose up' with the supplied docker-compose.yml file to start two containers (ElasticSearch and Kibana, both ver  7.13)
"""

parser = argparse.ArgumentParser(description="Generate FZxMedQA Dataset")
parser.add_argument(
    "chunk_size",
    type=int,
    nargs="?",
    default=100,
    help="number of tokens within a chunk",
)
parser.add_argument(
    "stride",
    type=int,
    nargs="?",
    default=50,
    help="size of stride window (to avoid excluding connected contexts)",
)
parser.add_argument(
    "topn",
    type=int,
    nargs="?",
    default=25,
    help="top n hits returned from ElasticSearch by a given query input",
)
parser.add_argument(
    "cache_dir",
    type=str,
    nargs="?",
    default="datamodules/datasets/",
    help="where to download temporary dataset files (relative dir from working directory)",
)
parser.add_argument(
    "output",
    type=str,
    nargs="?",
    default="datamodules/datasets/",
    help="where to output final datasets (relative dir from working directory)",
)
args = parser.parse_args()


def getDocChunks(article: str, chunkSize: int, stride: int):
    """
    takes in a FZ article as a string, 
    cuts the article into chunks, and 
    outputs a list of these chunks. 
    """
    #these lines ensure to remove e.g. wikipedia tags, linebreaks etc.
    wiki_tags = r'\[.*?\]'
    article = article.replace('\n', ' ').replace('\r', '')
    article = re.sub(wiki_tags, '', article)

    # splitting string into list of tokens based on spaces
    doc = article.split()

    # adding chunk to based while looping through article based on chunkSize and stride
    i = 0
    docChunks = []
    while i < len(doc):
        j = i + chunkSize
        tokens = doc[i:j]
        docChunks.append(" ".join(tokens))
        i += stride
    return docChunks

def ingest_all(data: dict, dateset_name: str):
    """
    ingesting all articles in a dict dataset in sense of 
    chunks with the aim of searching through the index
    to ranks chunks to a given input query.
    """
    # creates elastic search index
    es_create_index(dateset_name)

    # this loop runs the dataset, filtering the ~2k mapped questions 
    for key in tqdm(data.keys()):
        if data[key]['FZ_results']:
            # splits all articles to chunks
            for article in data[key]['FZ_results']:
                docs = getDocChunks(article['doc_context'], chunkSize=args.chunk_size, stride=args.stride)
                #ingesting chunks to elasticsearch index
                for doc in docs:
                    _ = es_ingest(dateset_name, article['title'], doc)


train_url = 'https://drive.google.com/uc?id=1K8Lu0rI2rK-WZFLxmuQ60mRNWIbSmiy2'
dev_url = 'https://drive.google.com/uc?id=16sJUgYCVwYSp5Zy35xW7NlUUBGhDNdWO'
test_url = 'https://drive.google.com/uc?id=1WZFwLpM_2RNHP2QE-JHlCm5mcb7I0FtN'

output = [
    str(os.path.join(args.cache_dir, "train.json")),
    str(os.path.join(args.cache_dir, "dev.json")),
    str(os.path.join(args.cache_dir, "test.json")),
]

gdown.cached_download(train_url, output[0], quiet=False)
gdown.cached_download(dev_url, output[1], quiet=False)
gdown.cached_download(test_url, output[2], quiet=False)

with open(output[0], "rb") as f:
    train = json.load(f)

with open(output[1], "rb") as f:
    val = json.load(f)

with open(output[2], "rb") as f:
    test = json.load(f)

datasets = [train, val, test]
ds_names = ["train", "val", "test"]

counter = 0
for ds_id, ds in enumerate(datasets):
    # offical HuggingFace datastructure
    # see: https://huggingface.co/docs/datasets/loading_datasets.html
    out = {
        'version': '0.0.1',
        'data': []
        }

    print("Ingesting all articles to ElasticSearch")
    ingest_all(ds, ds_names[ds_id])
    print("Finish ingesting all articles")

    # this loop creates a list of dicts appending samples to the dataset
    for key in tqdm(ds.keys()):
        if ds[key]['FZ_results']:
            counter+=1
            rank = 0
            q_id = key[1:]
            answer_options = [ds[key]['answer_options'][opt] for opt in ds[key]['answer_options'].keys()]

            # returning top n hits from es index based on question (query input)
            es_res = es_search(ds_names[ds_id], ds[key]['question'],args.topn)
            # es_res is sorted by BM25 score where the first instance has the highest score
            for hit in es_res['hits']:
                rank += 1
                 # creating a sample for each hit returned from elasticsearch
                out['data'].append({
                        'idx' : counter,
                        'question_id' : q_id,
                        'question' : ds[key]['question'],
                        'answer_choices' : answer_options,
                        'answer_idx' : answer_options.index(ds[key]['answer']),
                        'document' : hit['_source']['title'] + '. ' + hit['_source']['text'],
                        'rank' : rank #rank based on BM25
                    })
    #this line ensures to remove the elasticsearch index when finished
    es_remove_index(ds_names[ds_id])

    with open(os.path.join(args.output, ds_names[ds_id] + "_FZ-MedQA.json"), "w") as file:
        json.dump(out, file, indent=6)