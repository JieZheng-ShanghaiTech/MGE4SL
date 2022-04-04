import math
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

class SynlethDB(Data):
    def __init__(self, num_nodes, sl_data,nosl_data):
        num_nodes = num_nodes  
        num_edges = sl_data.shape[0]
        neg_num_edges = nosl_data.shape[0]
        feat_node_dim = 1
        feat_edge_dim = 1 
        self.x = torch.ones(num_nodes, feat_node_dim)
        self.y = torch.randint(0, 2, (num_nodes,)) 
        self.edge_index = torch.tensor(sl_data[['gene_a_encoder','gene_b_encoder']].T.values, dtype=torch.long)
        self.edge_attr = torch.ones(num_edges, feat_edge_dim)
        self.neg_edge_index = torch.tensor(nosl_data[['gene_a_encoder', 'gene_b_encoder']].T.values, dtype=torch.long)
        self.neg_edge_attr = torch.ones(neg_num_edges, feat_edge_dim)

#related knowledge graph
class SynlethDB_KG(Data):
    def __init__(self, kg_data, types):
        self.type = types
        num_nodes = 9872
        num_edges = kg_data.shape[0]
        feat_node_dim = 1 
        feat_edge_dim = 1 
        self.x = torch.ones(num_nodes, feat_node_dim) 
        self.y = torch.randint(0, 2, (num_nodes,)) 
        self.edge_index = torch.tensor(kg_data[['gene_a_encoder','gene_b_encoder']].T.values, dtype=torch.long)
        self.edge_attr = torch.tensor(kg_data[[self.type]].values, dtype = torch.long) 

#random negative sample  
def get_k_fold_data_random_neg(data, k = 10): 
    
    num_nodes = data.num_nodes
    
    row, col = data.edge_index
    num_edges = row.size(0)
    mask = row < col
    row, col = row[mask], col[mask]
    
    neg_row, neg_col = data.neg_edge_index
    neg_num_edges = neg_row.size(0)
    mask = neg_row < neg_col
    neg_row, neg_col = neg_row[mask], neg_col[mask]
    
    assert k > 1
    fold_size = num_edges // k  
    
    perm = torch.randperm(num_edges)
    row, col = row[perm], col[perm]
    
    neg_perm = torch.randperm(neg_num_edges)
    neg_row, neg_col = neg_row[neg_perm], neg_col[neg_perm]
    
    res_neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    
    res_neg_adj_mask = res_neg_adj_mask.triu(diagonal=1).to(torch.bool)
    res_neg_adj_mask[row, col] = 0
    res_neg_row, res_neg_col = res_neg_adj_mask.nonzero(as_tuple=False).t()
    

    for j in range(k):
        val_start = j *  fold_size
        val_end = (j+1) * fold_size
        if j == k - 1:
            val_row, val_col = row[val_start:], col[val_start:]
            train_row, train_col = row[:val_start], col[:val_start]
        else:
            val_row, val_col = row[val_start:val_end], col[val_start:val_end]
            train_row, train_col = torch.cat([row[:val_start],row[val_end:]], 0), torch.cat([col[:val_start],col[val_end:]], 0)
         
        # val
        data.val_pos_edge_index = torch.stack([val_row, val_col], dim=0)
        # train
        data.train_pos_edge_index = torch.stack([train_row, train_col], dim=0)
        
        add_val = data.val_pos_edge_index.shape[1]
        add_train = data.train_pos_edge_index.shape[1]
        perm = torch.randperm(res_neg_row.size(0))[:add_val+add_train]
        res_neg_row, res_neg_col = res_neg_row[perm], res_neg_col[perm]

        res_r, res_c = res_neg_row[:add_val], res_neg_col[:add_val]
        data.val_neg_edge_index = torch.stack([res_r, res_c], dim=0)
        
        res_r, res_c = res_neg_row[add_val:add_val+add_train], res_neg_col[add_val:add_val+add_train]
        data.train_neg_edge_index = torch.stack([res_r, res_c], dim=0)
    
        data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)
        data.train_neg_edge_index = to_undirected(data.train_neg_edge_index)
        yield data

def train_test_split_edges_kg(data, test_ratio=0.1):
    num_nodes = data.num_nodes
    row, col = data.edge_index
    
    data.edge_index = None
    num_edges = row.size(0)

    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]

    n_t = int(math.floor(test_ratio * num_edges))

    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]
    

    r, c = row[:n_t], col[:n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)

    r, c = row[n_t:], col[n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    
    data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)
    
    # Negative edges.
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0

    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    perm = torch.randperm(neg_row.size(0))[:num_edges]
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    row, col = neg_row[:n_t], neg_col[:n_t]
    data.test_neg_edge_index = torch.stack([row, col], dim=0)
    
    row, col = neg_row[n_t:], neg_col[n_t:]
    data.train_neg_edge_index = torch.stack([row, col], dim=0)
    
    data.train_neg_edge_index = to_undirected(data.train_neg_edge_index)
    return data


def construct_kg_sldb(data):
    combined_score_data = data[data['combined_score'] > 0]
    reactome_data = data[data['reactome'] > 0]
    corum_data = data[data['corum'] > 0]
    go_F_data = data[data['go_F'] > 0]
    go_C_data = data[data['go_C'] > 0]
    go_P_data = data[data['go_P'] > 0]
    kegg_data = data[data['kegg'] > 0]

    synlethdb_ppi = SynlethDB_KG(combined_score_data, 'combined_score')
    synlethdb_ppi = train_test_split_edges_kg(synlethdb_ppi, test_ratio=0)

    synlethdb_rea = SynlethDB_KG(reactome_data, 'reactome')
    synlethdb_rea = train_test_split_edges_kg(synlethdb_rea, test_ratio=0)

    synlethdb_cor = SynlethDB_KG(corum_data, 'corum')
    synlethdb_cor = train_test_split_edges_kg(synlethdb_cor, test_ratio=0)

    synlethdb_go_F = SynlethDB_KG(go_F_data, 'go_F')
    synlethdb_go_F = train_test_split_edges_kg(synlethdb_go_F, test_ratio=0)

    synlethdb_go_C = SynlethDB_KG(go_C_data, 'go_C')
    synlethdb_go_C = train_test_split_edges_kg(synlethdb_go_C, test_ratio=0)

    synlethdb_go_P = SynlethDB_KG(go_P_data, 'go_P')
    synlethdb_go_P = train_test_split_edges_kg(synlethdb_go_P, test_ratio=0)

    synlethdb_kegg = SynlethDB_KG(kegg_data, 'kegg')
    synlethdb_kegg = train_test_split_edges_kg(synlethdb_kegg, test_ratio=0)

    return synlethdb_ppi,synlethdb_rea,synlethdb_cor,synlethdb_go_F,synlethdb_go_C,synlethdb_go_P,synlethdb_kegg