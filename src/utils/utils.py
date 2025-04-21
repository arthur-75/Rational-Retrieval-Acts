from beir.retrieval.evaluation import EvaluateRetrieval
#from neural_che.neural_cherche import  models, utils, retrieve
from scipy.sparse import csr_matrix
import numpy as np
import torch


def get_torch_sparse(doc_to_transform,row_sums=[], device=None,apply_log=False,keep_all=False,dtype=torch.float32):

    """
    Perform operations on sparse tensors and convert to dense just before applying F.softplus.
    """
    # Set device to GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
    
    # Stack sparse matrices and convert to a PyTorch sparse tensor
    if type(doc_to_transform )is dict:
        scipy_sparse_matrix = vstack(list(doc_to_transform.values()))
        coo = scipy_sparse_matrix.tocoo().T
    else :
        scipy_sparse_matrix = (doc_to_transform)
        coo = scipy_sparse_matrix.tocoo()
    # Convert to PyTorch sparse tensor more efficiently
    indices = torch.tensor([coo.row, coo.col], dtype=torch.int, device=device)    
    values = torch.tensor(coo.data, dtype=dtype, device=device)
    if apply_log: 
        values=torch.log1p(torch.relu(values))

    sparse_tensor = torch.sparse_coo_tensor(indices, values, coo.shape, device=device,dtype=dtype)
    # Get keys as a list (for output)
    # Row sums: sum over the message/world dimension using the sparse tensor
    if  len(row_sums)==0:
        row_sums = torch.sparse.sum(sparse_tensor, dim=1)
        row_sums= row_sums.indices()[0]
    # Apply mask to keep rows where sum is non-zero (filtering out zero rows)
 
    # Create the new filtered sparse tensor
    del coo,indices,values,scipy_sparse_matrix
    # keep only row with sum not equal to 0
    if not keep_all:
        sparse_tensor = sparse_tensor.index_select(0, row_sums)

    return sparse_tensor, row_sums 






def metreic(scores_rsa,qrels):
    ndcg, map_, recall, p = EvaluateRetrieval.evaluate(qrels, scores_rsa, [1, 10, 100, 1000])
    metrics = {
        "NDCG@1": ndcg["NDCG@1"],
        "NDCG@10": ndcg["NDCG@10"],
        "NDCG@100": ndcg["NDCG@100"],
        "Recall@1": recall["Recall@1"],
        "Recall@10": recall["Recall@10"],
        "Recall@100": recall["Recall@100"],
    }
    return metrics

def get_score(sparse_torch,input_ids,doc_ids,query_ids,qrels,Vd=None,Ut=None,top_k=100):
    #input_ids=sparse_model.model.tokenizer(queries,padding=True, truncation=True, return_tensors='pt', max_length=512)['input_ids']
    doc_spa=get_coo(sparse_torch)
    print(doc_spa.shape)
    results={}
    for start_idx in (range(len(input_ids))):
            qid = query_ids[start_idx]
            query_tokens = input_ids[start_idx]
            scores = np.asarray(doc_spa[query_tokens, :].sum(axis=0)).squeeze(0)

            if "None" not in str(Ut):  scores+= np.asarray(Vd* Ut[input_ids[start_idx]].sum())
            top_k_ind = np.argpartition(scores, -top_k)[-top_k:]
            results[qid] = {doc_ids[pid]: float(scores[pid]) for pid in top_k_ind[::-1] if doc_ids[pid] != qid}
    # Ensure it's a valid sparse tensor
    metrics= metreic(results,qrels),results
    return metrics
def get_coo(rsa_mat):
    # Extract indices and values
    coo_indices = rsa_mat.coalesce().indices()  # shape: (2, nnz)
    coo_values = rsa_mat.coalesce().values()    # shape: (nnz,)
    shape = rsa_mat.size()           # (rows, cols)

    row = coo_indices[0].numpy()
    col = coo_indices[1].numpy()
    data = coo_values.numpy()

    # Create a SciPy coo_matrix
    scipy_coo = csr_matrix((data, (row, col)), shape=shape, dtype=float)#.T
    return scipy_coo