
from beir.retrieval.search.sparse import SparseSearch
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval


def get_model(model_path = "BeIR/sparta-msmarco-distilbert-base-v1"):
    
    sparse_model = SparseSearch(models.SPARTA(model_path,max_length=512, add_special_tokens=False), batch_size=64)

    return sparse_model


def encode(sparse_model,corpus,queries,queries_ids,qrels,result_without_RSA=False):
    input_ids=sparse_model.model.tokenizer(queries_ids, add_special_tokens=False)['input_ids']
    retriever = EvaluateRetrieval(sparse_model)
    results = retriever.retrieve(corpus, queries)
    
    if result_without_RSA:
        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
        print(ndcg)
    return sparse_model.sparse_matrix,input_ids


