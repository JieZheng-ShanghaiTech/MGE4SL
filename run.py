import argparse
import torch

from utils import run_experiment
from models import MultiGraphEnsembleFC, MultiGraphEnsembleFC_SUM, MultiGraphEnsembleCNN, MultiGraphEnsembleWeightFC
if __name__ == '__main__':
    LR = 0.01
    EPOCHS = 20
    data_path = './data/human_sl_encoder.csv'
    
    model = MultiGraphEnsembleFC(n_graph=8, node_emb_dim=16, sl_input_dim=1, kg_input_dim=1)
    #model = MultiGraphEnsembleFC_SUM(n_graph=8, node_emb_dim=16, sl_input_dim=1, kg_input_dim=1)
    #model = MultiGraphEnsembleCNN(n_graph=8, node_emb_dim=16, sl_input_dim=1, kg_input_dim=1)
    #model = MultiGraphEnsembleWeightFC(n_graph=7, node_emb_dim=16, sl_input_dim=1, kg_input_dim=1)

    run_experiment(data_path, model, "MGE_CNN", epochs= EPOCHS, lr=LR)