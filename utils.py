import torch
import pandas as pd
import numpy as np
import os

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
from data_prepare import SynlethDB, get_k_fold_data_random_neg, construct_kg_sldb

from scipy import stats
from scipy.stats import t
import torch.nn.functional as F

def cal_confidence_interval(data, confidence=0.95):
    data = 1.0*np.array(data)
    n = len(data)
    sample_mean = np.mean(data)
    se = stats.sem(data)
    t_ci = t.ppf((1+confidence)/2., n-1)  # T value of Confidence Interval
    bound = se * t_ci
    return sample_mean, bound

def get_link_labels(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

def evaluate(y_true, y_score, pos_threshold = 0.8):
    auc_test = roc_auc_score(y_true, y_score)
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    aupr_test = auc(recall, precision)
    f1_test = f1_score(y_true, y_score > pos_threshold)
    return auc_test, aupr_test, f1_test

def train(model, optimizer, synlethdb_sl, synlethdb_ppi, synlethdb_rea, synlethdb_cor, synlethdb_go_F, \
            synlethdb_go_C, synlethdb_go_P, synlethdb_kegg):
    model.train()
    optimizer.zero_grad()
    
    pos_edge_index = synlethdb_sl.train_pos_edge_index
    neg_edge_index = synlethdb_sl.train_neg_edge_index


    link_pred = model(synlethdb_sl.x, pos_edge_index, neg_edge_index, synlethdb_ppi.train_pos_edge_index, \
      synlethdb_rea.train_pos_edge_index, synlethdb_cor.train_pos_edge_index, synlethdb_go_F.train_pos_edge_index,\
      synlethdb_go_C.train_pos_edge_index, synlethdb_go_P.train_pos_edge_index, synlethdb_kegg.train_pos_edge_index
     )

    link_labels = get_link_labels(pos_edge_index, neg_edge_index)
    loss = F.binary_cross_entropy(link_pred, link_labels)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test(model, synlethdb_sl, synlethdb_ppi, synlethdb_rea, synlethdb_cor, synlethdb_go_F, \
            synlethdb_go_C, synlethdb_go_P, synlethdb_kegg):
    model.eval()
    
    pos_edge_index = synlethdb_sl.val_pos_edge_index
    neg_edge_index = synlethdb_sl.val_neg_edge_index
    
    perfs = [] 
    link_pred = model(synlethdb_sl.x, pos_edge_index, neg_edge_index, synlethdb_ppi.train_pos_edge_index, \
      synlethdb_rea.train_pos_edge_index, synlethdb_cor.train_pos_edge_index, synlethdb_go_F.train_pos_edge_index,\
      synlethdb_go_C.train_pos_edge_index, synlethdb_go_P.train_pos_edge_index, synlethdb_kegg.train_pos_edge_index
     )
    
    link_labels = get_link_labels(pos_edge_index, neg_edge_index)
    
    auc_test, aupr_test, f1_test = evaluate(link_labels.cpu(), link_pred.cpu())
    perfs.extend([auc_test, aupr_test, f1_test])

    return perfs

def run_experiment(data_path, model, model_name, epochs, lr=0.01):
    if not os.path.exists("models"):
        os.makedirs("models")
    data = pd.read_csv(data_path)
    sl_data = data[data['sl'] == 1]
    nosl_data = data[data['sl'] != 1]
    
    with open("./data/genes_list.txt") as f:
        genes_list = f.readlines()
    num_nodes = len(genes_list)
    
    synlethdb = SynlethDB(num_nodes, sl_data, nosl_data)
    k_fold = get_k_fold_data_random_neg(synlethdb, k = 5)

    synlethdb_ppi,synlethdb_rea,synlethdb_cor,synlethdb_go_F,\
    synlethdb_go_C,synlethdb_go_P,synlethdb_kegg = construct_kg_sldb(data)
    print("data prepare finished!")
    k_val_best_auc = []
    k_val_best_aupr = []
    k_val_best_f1 = []

    k = 0 
    for k_data in k_fold:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        explr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

        print('start one fold:')
        best_score_sum  = 0

        for epoch in range(0, epochs):
            train_loss = train(model, optimizer, k_data, synlethdb_ppi,synlethdb_rea,synlethdb_cor,synlethdb_go_F,\
                            synlethdb_go_C,synlethdb_go_P,synlethdb_kegg)
            val_perf = test(model, k_data, synlethdb_ppi,synlethdb_rea,synlethdb_cor,synlethdb_go_F,\
                            synlethdb_go_C,synlethdb_go_P,synlethdb_kegg)
            explr_scheduler.step()
            
            score_sum = np.array(val_perf).sum()
            if best_score_sum < score_sum:
                best_score_sum = score_sum
                torch.save(model, './models/%s_%d'%(model_name, k) + '.pkl')
                best_val_perf_auc = val_perf[0]
                best_val_perf_aupr = val_perf[1]
                best_val_perf_f1 = val_perf[2]
                
            log = 'Epoch: {:03d}, Loss: {:.4f}, \
                Val_AUC: {:.4f}, Val_AUPR:{:.4f}, Val_F1:{:.4f},'
            print(log.format(epoch, train_loss, val_perf[0], val_perf[1], val_perf[2]))
        
        k += 1
        k_val_best_auc.append(best_val_perf_auc)
        k_val_best_aupr.append(best_val_perf_aupr)
        k_val_best_f1.append(best_val_perf_f1)
    print('auc:', cal_confidence_interval(k_val_best_auc))
    print('aupr:', cal_confidence_interval(k_val_best_aupr))
    print('f1:', cal_confidence_interval(k_val_best_f1))