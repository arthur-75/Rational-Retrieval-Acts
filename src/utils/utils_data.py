from beir import util
from beir.datasets.data_loader import GenericDataLoader

def get_data(data_set,data_path="data"):

    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(data_set)
    data_path = util.download_and_unzip(url, data_path)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    #queries_ids = list(queries.keys())
    #queries= list(queries.values())
    #documents = [[f"{doc['title']} ,{doc['text']}"] for doc in corpus.values()]
    #document_ids= list(corpus.keys())

    return corpus, queries, qrels 


