import matplotlib.pyplot as plt
import torch
from code.ResultSaving import ResultSaving
from code.EvaluateClustering import EvaluateClustering

from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, matthews_corrcoef, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, auc, f1_score, precision_score, recall_score
import torch.nn.functional as F
import itertools


#--------------- Graph Bert Learning Convergence --------------

dataset_name = 'ppi'

accuracy = []
tp = []
tn = []
fp = []
fn = []
prf = []
sensitivity = []
specificity = []
mcc = []
roc = []
pr = []

if 1:
    residual_type = 'graph_raw'
    diffusion_type = 'sum'
    hidden_layers = 2
    #depth_list = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50]#, 2, 3, 4, 5, 6, 9, 19, 29, 39, 49]
    result_obj = ResultSaving('', '')
    result_obj.result_destination_folder_path = './result/GraphBert/'
    best_score = {}

    depth_result_dict = {}
    y_true = []
    y_pred = []
    depth = 2
    

    result_obj.result_destination_file_name = dataset_name + '_' + str(depth)
    #print(result_obj.result_destination_file_name)
    depth_result_dict[depth] = result_obj.load()
    #print(depth_result_dict)

    x = range(120)

    test_acc = [depth_result_dict[depth][i]['acc_test'] for i in x] 
    #print(test_acc, '\t' , len(test_acc)) 
    
    bestEpoch = 0
    for i in range(1,len(test_acc)):
      if test_acc[bestEpoch] < test_acc[i]:
        bestEpoch = i
    print('best epoch : ',bestEpoch) 
    test_acc_data = depth_result_dict[depth][bestEpoch]['test_acc_data']
    y_true = test_acc_data['true_y'].tolist()
    y_pred = test_acc_data['pred_y'].tolist()
    
    test_op = depth_result_dict[depth][bestEpoch]['test_op']#.tolist()
    #print(test_op[0], '\t' , len(test_op))
    probs = F.softmax(test_op, dim=1)
    probs = probs.max(1)[1]   
    probs = probs.tolist()
    best_score[depth] = max(test_acc)

    ############################ Evaluation Metrices #######################
    '''# accuracy: (tp + tn) / (p + n)
    _accuracy = accuracy_score(y_true, y_pred)
    print('Accuracy: %f' % _accuracy)
    
    # specificity
    
    # mcc score
    
    # precision-recall curve
    # prediction values 
    
    # precision tp / (tp + fp)
    precision = precision_score(y_true, y_pred)
    print('Precision: %f' % precision)
    
    # recall: tp / (tp + fn)
    recall = recall_score(y_true, y_pred)
    print('Recall: %f' % recall)
    
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_true, y_pred)
    print('F1 score: %f' % f1)
    
    # ROC AUC
    ## AUC -> pass probability value instead of y_pred
    auc = roc_auc_score(y_true, top_p)
    print('ROC AUC: %f' % auc)

    #print('y_test score = ',y_test)'''
    print('\n\nEvaluation Metrices :\n')
    
    y_test = y_true
    y_pred_nn = probs
    y_pred_nn1 = y_pred
    acc = accuracy_score(y_test, y_pred_nn1)
    accuracy.append(acc)
    tn1, fp1, fn1, tp1 = confusion_matrix(y_test, y_pred_nn1).ravel()
    sens = tp1/(tp1+fn1)
    spec = tn1/(tn1+fp1)
    prf1 = precision_recall_fscore_support(y_test, y_pred_nn1)
    tp.append(tp1)
    tn.append(tn1)
    fp.append(fp1)
    fn.append(fn1)
    prf.append(prf1)
    sensitivity.append(sens)
    specificity.append(spec)
    mathew = matthews_corrcoef(y_test, y_pred_nn1)
    mcc.append(mathew)
    auroc = roc_auc_score(y_test, y_pred_nn)
    roc.append(auroc)
    aupr = average_precision_score(y_test, y_pred_nn)
    pr.append(aupr)
    
    
    print('accuracy',accuracy)
    print('sensitivity',sensitivity)
    print('specificity',specificity)
    print('prf',prf)
    print('tp tn fp fn', tp, tn, fp, fn)
    print('mcc', mcc)
    print('roc', roc)
    print('pr', pr)
