import numpy as np
import torch

from code.DatasetLoader import DatasetLoader
from code.MethodWLNodeColoring import MethodWLNodeColoring
from code.MethodGraphBatching import MethodGraphBatching
from code.MethodHopDistance import MethodHopDistance
from code.ResultSaving import ResultSaving
from code.Settings import Settings

#--- ppi, c.elegan ----

dataset_name = 'ppi'

np.random.seed(1)
torch.manual_seed(1)

#---- cora-small is for debuging only ----
if dataset_name == 'ppi':
    nclass = 2
    nfeature = 2048
    ngraph = 16324 #26324 

    
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


elif dataset_name == 'human':
    nclass = 2
    nfeature = 2048
    ngraph = 56919  #27920


#---- Step 1: WL based graph coloring ----
if 1:
    print('************ Start ************')
    print('WL, dataset: ' + dataset_name)
    # ---- objection initialization setction ---------------
    data_obj = DatasetLoader()
    data_obj.dataset_source_folder_path = './data/' + dataset_name + '/'
    data_obj.dataset_name = dataset_name

    method_obj = MethodWLNodeColoring()

    result_obj = ResultSaving()
    result_obj.result_destination_folder_path = './result/WL/'
    result_obj.result_destination_file_name = dataset_name

    setting_obj = Settings()

    evaluate_obj = None
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.load_run_save_evaluate()
    # ------------------------------------------------------

    print('************ Finish ************')
#------------------------------------

#---- Step 2: intimacy calculation and subgraph batching ----
if 1:
    for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:#, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        print('************ Start ************')
        print('Subgraph Batching, dataset: ' + dataset_name + ', k: ' + str(k))
        # ---- objection initialization setction ---------------
        data_obj = DatasetLoader()
        data_obj.dataset_source_folder_path = './data/' + dataset_name + '/'
        data_obj.dataset_name = dataset_name
        data_obj.compute_s = True

        method_obj = MethodGraphBatching()
        method_obj.k = k

        result_obj = ResultSaving()
        result_obj.result_destination_folder_path = './result/Batch/'
        result_obj.result_destination_file_name = dataset_name + '_' + str(k)

        setting_obj = Settings()

        evaluate_obj = None
        # ------------------------------------------------------

        # ---- running section ---------------------------------
        setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
        setting_obj.load_run_save_evaluate()
        # ------------------------------------------------------

        print('************ Finish ************')
#------------------------------------

#---- Step 3: Shortest path: hop distance among nodes ----
if 1:
    for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        print('************ Start ************')
        print('HopDistance, dataset: ' + dataset_name + ', k: ' + str(k))
        # ---- objection initialization setction ---------------
        data_obj = DatasetLoader()
        data_obj.dataset_source_folder_path = './data/' + dataset_name + '/'
        data_obj.dataset_name = dataset_name

        method_obj = MethodHopDistance()
        method_obj.k = k
        method_obj.dataset_name = dataset_name

        result_obj = ResultSaving()
        result_obj.result_destination_folder_path = './result/Hop/'
        result_obj.result_destination_file_name = 'hop_' + dataset_name + '_' + str(k)

        setting_obj = Settings()

        evaluate_obj = None
        # ------------------------------------------------------

        # ---- running section ---------------------------------
        setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
        setting_obj.load_run_save_evaluate()
        # ------------------------------------------------------

        print('************ Finish ************')
#------------------------------------
