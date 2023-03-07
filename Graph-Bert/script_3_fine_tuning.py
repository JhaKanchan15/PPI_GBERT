from code.DatasetLoader import DatasetLoader
from code.MethodBertComp import GraphBertConfig
from code.MethodGraphBertNodeClassification import MethodGraphBertNodeClassification
from code.ResultSaving import ResultSaving
from code.Settings import Settings
import numpy as np
import torch


#---- 'cora' , 'citeseer', 'pubmed' ----

dataset_name = 'ppi'

np.random.seed(1)
torch.manual_seed(1)

#---- cora-small is for debuging only ----
if dataset_name == 'ppi':
    nclass = 2
    nfeature = 2048
    ngraph = 16324

elif dataset_name == 'human':
    nclass = 2
    nfeature = 2048
    ngraph = 56919  #27920 --> modified this 19/02/23

elif dataset_name == 'c.elegan':
    nclass = 2
    nfeature = 2048
    ngraph = 4548

elif dataset_name == 'drosophila':
    nclass = 2
    nfeature = 2048
    ngraph = 26602  

elif dataset_name == 'e.coli':
    nclass = 2
    nfeature = 2048
    ngraph = 9607


#---- Fine-Tuning Task 1: Graph Bert Node Classification (Cora, Citeseer, and Pubmed) ----
if 1:
    #---- hyper-parameters ----
    if dataset_name == 'ppi':
        lr = 0.001
        k = 7
        max_epoch = 250 # 500 ---- do an early stop when necessary ----
    
    elif dataset_name == 'human':
        lr = 0.001
        k = 7
        max_epoch = 120
    
    elif dataset_name == 'c.elegan':
        lr = 0.001
        k = 7
        max_epoch = 120 # 150 ---- do an early stop when necessary ----
    
    elif dataset_name == 'e.coli':
        lr = 0.001
        k = 7
        max_epoch = 120
        
    elif dataset_name == 'drosophila':
        lr = 0.001
        k = 7
        max_epoch = 120 # 150 ---- do an early stop when necessary ----
                 
    

    x_size = nfeature
    hidden_size = intermediate_size = 32
    num_attention_heads = 2
    num_hidden_layers = 2
    y_size = nclass
    graph_size = ngraph
    residual_type = 'graph_raw'
    # --------------------------

    print('************ Start ************')
    print('GrapBert, dataset: ' + dataset_name + ', residual: ' + residual_type + ', k: ' + str(k) + ', hidden dimension: ' + str(hidden_size) +', hidden layer: ' + str(num_hidden_layers) + ', attention head: ' + str(num_attention_heads))
    # ---- objection initialization setction ---------------
    data_obj = DatasetLoader()
    data_obj.dataset_source_folder_path = './data/' + dataset_name + '/'
    data_obj.dataset_name = dataset_name
    data_obj.k = k
    data_obj.load_all_tag = True

    bert_config = GraphBertConfig(residual_type = residual_type, k=k, x_size=nfeature, y_size=y_size, hidden_size=hidden_size, intermediate_size=intermediate_size, num_attention_heads=num_attention_heads, num_hidden_layers=num_hidden_layers)
    method_obj = MethodGraphBertNodeClassification(bert_config)
    #---- set to false to run faster ----
    method_obj.spy_tag = True
    method_obj.max_epoch = max_epoch
    method_obj.lr = lr

    result_obj = ResultSaving()
    result_obj.result_destination_folder_path = './result/GraphBert/'
    result_obj.result_destination_file_name = dataset_name + '_' + str(num_hidden_layers)

    setting_obj = Settings()

    evaluate_obj = None
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.load_run_save_evaluate()
    # ------------------------------------------------------


    method_obj.save_pretrained('./result/PreTrained_GraphBert/' + dataset_name + '/node_classification_complete_model/')
    print('************ Finish ************')
#------------------------------------


