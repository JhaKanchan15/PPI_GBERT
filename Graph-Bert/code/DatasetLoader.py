'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from code.base_class.dataset import dataset
import torch
import numpy as np
import scipy.sparse as sp
from numpy.linalg import inv
import pickle
from itertools import chain

class DatasetLoader(dataset):
    c = 0.15
    k = 5
    data = None
    batch_size = None

    dataset_source_folder_path = None
    dataset_name = None

    load_all_tag = False
    compute_s = False

    def __init__(self, seed=None, dName=None, dDescription=None):
        super(DatasetLoader, self).__init__(dName, dDescription)

    def load_hop_wl_batch(self):
        print('Load WL Dictionary')
        f = open('./result/WL/' + self.dataset_name, 'rb')
        wl_dict = pickle.load(f)
        f.close()

        print('Load Hop Distance Dictionary')
        f = open('./result/Hop/hop_' + self.dataset_name + '_' + str(self.k), 'rb')
        hop_dict = pickle.load(f)
        f.close()

        print('Load Subgraph Batches')
        f = open('./result/Batch/' + self.dataset_name + '_' + str(self.k), 'rb')
        batch_dict = pickle.load(f)
        f.close()

        return hop_dict, wl_dict, batch_dict

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def adj_normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
        return mx

    def accuracy(self, output, labels):
        preds = output.max(1)[1].type_as(labels)
        correct = preds.eq(labels).double()
        correct = correct.sum()
        return correct / len(labels)

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                        enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                 dtype=np.int32)
        return labels_onehot

    def load(self):
        """Load citation network dataset (cora only for now)"""
        print('Loading {} dataset...'.format(self.dataset_name))

        idx_features_labels = np.genfromtxt("{}/node".format(self.dataset_source_folder_path), dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)

        one_hot_labels = self.encode_onehot(idx_features_labels[:, -1])

        # build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        #print('idx = ',idx,'\n\n', idx[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        #print('idx_map', idx,'\n\n')
        index_id_map = {i: j for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}/link".format(self.dataset_source_folder_path),
                                        dtype=np.int32)
        #print('\n\n edges_unordered = ',edges_unordered)
        my_arr = list(map(idx_map.get, edges_unordered.flatten()))
        res_arr = [0 if i is None else i for i in my_arr]
        #print('my_arr = ',my_arr,'\n\n res_arr = ',res_arr,'\n\n')
        
        edges = np.array(res_arr,dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(one_hot_labels.shape[0], one_hot_labels.shape[0]),
                            dtype=np.float32)

        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        eigen_adj = None
        if self.compute_s:
            eigen_adj = self.c * inv((sp.eye(adj.shape[0]) - (1 - self.c) * self.adj_normalize(adj)).toarray())

        norm_adj = self.adj_normalize(adj + sp.eye(adj.shape[0]))
        
        
        ########### human data division ##########
        '''
        #5th cross validation
        if self.dataset_name == 'human':
            idx_train = list(range(16740)) + list(range(22320, 27900))
            idx_test = range(16740, 22320)
            idx_val = range(16740, 22320)
        
        #4th cross validation
        if self.dataset_name == 'human':
            idx_train = list(range(11160)) + list(range(16740, 27900))
            idx_test = range(11160, 16740)
            idx_val = range(11160, 16740)
        
        #3rd cross validation
        if self.dataset_name == 'human':
            idx_train = list(range(5580)) + list(range(11160, 27900))
            idx_test = range(5580, 11160)
            idx_val = range(5580, 11160)
        
        
        #2nd cross validation
        if self.dataset_name == 'human':
            idx_train = range(5580, 27900)
            idx_test = range(5580)
            idx_val = range(5580)'''
        
        #1st cross validation
        if self.dataset_name == 'human':
            idx_train = range(22300)
            idx_test = range(22300, 27900)
            idx_val = range(22300, 27900)
         
        
        ########### e.coli data division ##########
        '''
        #5th cross validation
        if self.dataset_name == 'e.coli':
            idx_train = list(range(5764)) + list(range(7686, 9607))
            idx_test = range(5764, 7686)
            idx_val = range(5764, 7686)
        
        #4th cross validation
        if self.dataset_name == 'e.coli':
            idx_train = list(range(3843)) + list(range(5764, 9607))
            idx_test = range(3843, 5764)
            idx_val = range(3843, 5764)
        
        #3rd cross validation
        if self.dataset_name == 'e.coli':
            idx_train = list(range(1921)) + list(range(3843, 9607))
            idx_test = range(1921, 3843)
            idx_val = range(1921, 3843)
        
        #2nd cross validation
        if self.dataset_name == 'e.coli':
            idx_train = range(1921, 9607)
            idx_test = range(1921)
            idx_val = range(1921)'''
        
        #1st cross validation
        if self.dataset_name == 'e.coli':
            idx_train = range(7686)
            idx_test = range(7686, 9607)
            idx_val = range(7686, 9607)
  
        
        ########### drosophila data division ##########
        '''
        # 5th cross validation
        if self.dataset_name == 'drosophila':
            idx_train = list(range(15960)) + list(range(21280, 26600))
            idx_test = range(15960, 21280)
            idx_val = range(15960, 21280)
        
        # 4th cross validation
        if self.dataset_name == 'drosophila':
            idx_train = list(range(10640)) + list(range(15960, 26600))
            idx_test = range(10640, 15960)
            idx_val = range(10640, 15960)
        
        # 3rd cross validation
        if self.dataset_name == 'drosophila':
            idx_train = list(range(5320)) + list(range(10640, 26600))
            idx_test = range(5320, 10640)
            idx_val = range(5320, 10640)
        
        # 2nd cross validation
        if self.dataset_name == 'drosophila':
            idx_train = range(5320, 21280)
            idx_test = range(5320)
            idx_val = range(5320)'''
        
        # 1st cross validation
        if self.dataset_name == 'drosophila':
            idx_train = range(21280)
            idx_test = range(21280, 26600)
            idx_val = range(21280, 26600)
            
       
        ########### C.elegan data division ##########
        '''
        # 5th cross validation 
        elif self.dataset_name == 'c.elegan':
            idx_train = list(range(2724)) + list(range(3640, 4540))
            idx_test = range(2724,3640)
            idx_val = range(2724,3640)
            #features = self.normalize(features)
        
        # 4th cross validation 
        elif self.dataset_name == 'c.elegan':
            idx_train = list(range(1816)) + list(range(2724, 4540))
            idx_test = range(1816,2724)
            idx_val = range(1816,2724)
            #features = self.normalize(features)
        
        # 3rd cross validation 
        elif self.dataset_name == 'c.elegan':
            idx_train = list(range(908)) + list(range(1816, 4540))
            idx_test = range(908,1816)
            idx_val = range(908,1816)
            #features = self.normalize(features)
        
        # 2nd cross validation 
        elif self.dataset_name == 'c.elegan':
            idx_train = range(908, 4540)
            idx_test = range(908)
            idx_val = range(908)
            #features = self.normalize(features)'''
        
        # 1st cross validation - DONE
        if self.dataset_name == 'c.elegan':
            idx_train = range(3640)
            idx_test = range(3640, 4540)
            idx_val = range(3640, 4540)
            #features = self.normalize(features)
        
        
        ########## PPI (Hprd) dataset division #################
        
        '''
        # 5th cross validation for Hprd 
        elif self.dataset_name == 'ppi':
            idx_train = list(range(15795)) +list(range(21059,26322))
            idx_test = range(15795,21059)
            idx_val = range(15795,21059)
            print(type(idx_train), '\n',type(idx_test))
        
        
        # 4th cross validation for Hprd 
        elif self.dataset_name == 'ppi':
            idx_train = list(range(10530)) +list(range(15795,26322))
            idx_test = range(10530,15795)
            idx_val = range(10530,15795)
            print(type(idx_train), '\n',type(idx_test))
        
        # 3rd cross validation for Hprd 
        elif self.dataset_name == 'ppi':
            idx_train = list(range(5265)) +list(range(10530,26322))
            idx_test = range(5265,10530)
            idx_val = range(5265,10530)
            print(type(idx_train), '\n',type(idx_test))
        
        # 2nd cross validation for Hprd 
        elif self.dataset_name == 'ppi':
            idx_train = range(5265, 26322)
            idx_test = range(5265)
            idx_val = range(5265)'''
        
        # 1st cross validation for Hprd - DONE
        if self.dataset_name == 'ppi':
            idx_train = range(10000)
            idx_test = range(10000, 13000)
            idx_val = range(13000, 16324)


        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(one_hot_labels)[1])
        adj = self.sparse_mx_to_torch_sparse_tensor(norm_adj)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        if self.load_all_tag:
            hop_dict, wl_dict, batch_dict = self.load_hop_wl_batch()
            raw_feature_list = []
            role_ids_list = []
            position_ids_list = []
            hop_ids_list = []
            for node in idx:
                node_index = idx_map[node]
                neighbors_list = batch_dict[node]

                raw_feature = [features[node_index].tolist()]
                role_ids = [wl_dict[node]]
                position_ids = range(len(neighbors_list) + 1)
                hop_ids = [0]
                for neighbor, intimacy_score in neighbors_list:
                    if idx_map[neighbor] != "None":
                     neighbor_index = idx_map[neighbor]
                     raw_feature.append(features[neighbor_index].tolist())
                     role_ids.append(wl_dict[neighbor])
                     if hop_dict[node] != "None" and hop_dict[node][neighbor] != "None":
                      if neighbor in hop_dict[node]:
                          hop_ids.append(hop_dict[node][neighbor])
                      else:
                          hop_ids.append(99)
                raw_feature_list.append(raw_feature)
                role_ids_list.append(role_ids)
                position_ids_list.append(position_ids)
                hop_ids_list.append(hop_ids)
            raw_embeddings = torch.FloatTensor(raw_feature_list)
            wl_embedding = torch.LongTensor(role_ids_list)
            hop_embeddings = torch.LongTensor(hop_ids_list)
            int_embeddings = torch.LongTensor(position_ids_list)
        else:
            raw_embeddings, wl_embedding, hop_embeddings, int_embeddings = None, None, None, None

        return {'X': features, 'A': adj, 'S': eigen_adj, 'index_id_map': index_id_map, 'edges': edges_unordered, 'raw_embeddings': raw_embeddings, 'wl_embedding': wl_embedding, 'hop_embeddings': hop_embeddings, 'int_embeddings': int_embeddings, 'y': labels, 'idx': idx, 'idx_train': idx_train, 'idx_test': idx_test, 'idx_val': idx_val}
