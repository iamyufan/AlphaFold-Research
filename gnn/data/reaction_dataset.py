import pickle
import os
import numpy as np
import scipy.sparse as sp
from collections import Counter, defaultdict
from sklearn.preprocessing import StandardScaler


class ReactionDataset:
    def __init__(self, path):
        self.path = path
        self.nodes = self.load_nodes()
        self.links = self.load_links()
        self.labels_train = self.load_labels('label.dat')
        self.labels_test = self.load_labels('label.dat.test')

    def load_nodes(self):
        """
        return nodes dict
            total: total number of nodes
            count: a dict of int, number of nodes for each type
            attr: a dict of np.array (or None), attribute matrices for each type of nodes
            shift: node_id shift for each type. You can get the id range of a type by 
                        [ shift[node_type], shift[node_type]+count[node_type] )
        """
        nodes = {'total': 0, 'count': Counter(), 'attr': {}, 'shift': {}}
        
        # load node.pkl
        data = pickle.load(open(os.path.join(self.path, 'node.pkl'), 'rb'))
        
        nodes['total'] = data.shape[0]
        nodes['count'] = Counter(data['node_type_id'])
        
        e_data = data[data['node_type_id'] == 0]
        m_data = data[data['node_type_id'] == 1]
        
        # attr of enzyme
        e_attr = np.stack(e_data['node_feature'])
        scaler = StandardScaler()
        scaler.fit(e_attr)
        e_attr = scaler.transform(e_attr)

        # attr of molecule
        m_attr = np.stack(m_data['node_feature'])
        scaler = StandardScaler()
        scaler.fit(m_attr)
        m_attr = scaler.transform(m_attr)
        
        nodes['attr'] = {0:e_attr, 1:m_attr}
        nodes['shift'] = {0:0, 1:len(e_data)}

        return nodes

    def load_links(self):
        """
        return links dict
            total: total number of links
            count: a dict of int, number of links for each type
            meta: a dict of tuple, explaining the link type is from what type of node to what type of node
            data: a dict of sparse matrices, each link type with one matrix. Shapes are all (nodes['total'], nodes['total'])
        """
        links = {'total': 0, 'count': Counter(), 'meta': {},
                 'data': defaultdict(list)}
        with open(os.path.join(self.path, 'link.dat'), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                h_id, t_id, r_id, link_weight = int(
                    th[0]), int(th[1]), int(th[2]), float(th[3])
                if r_id not in links['meta']:
                    h_type = self.get_node_type(h_id)
                    t_type = self.get_node_type(t_id)
                    links['meta'][r_id] = (h_type, t_type)
                links['data'][r_id].append((h_id, t_id, link_weight))
                links['count'][r_id] += 1
                links['total'] += 1
        new_data = {}
        for r_id in links['data']:
            new_data[r_id] = self.list_to_sp_mat(links['data'][r_id])
        links['data'] = new_data
        return links

    def load_labels(self, name):
        """
        return labels dict
            num_labels: total number of labels
            total: total number of labeled data
            count: number of labeled data for each node type
            data: a numpy matrix with shape (self.nodes['total'], self.labels['num_labels'])
            mask: to indicate if that node is labeled, if False, that line of data is masked
        """
        labels = {'num_labels': 0, 'total': 0,
                  'count': Counter(), 'data': None, 'mask': None}
        nl = 1
        
        if nl == 1:
            mask = np.zeros(self.nodes['total'], dtype=bool)
            data = [0.0 for i in range(self.nodes['total'])]
            with open(os.path.join(self.path, name), 'r', encoding='utf-8') as f:
                for line in f:
                    th = line.split('\t')
                    node_id, node_type, node_label = int(th[0]), int(th[1]), float(th[2])
                    mask[node_id] = True
                    data[node_id] = node_label
                    labels['count'][node_type] += 1
                    labels['total'] += 1
            labels['num_labels'] = nl
            labels['data'] = np.array(data)
            labels['mask'] = mask
        
        elif nl == 2:
            mask = np.zeros(self.nodes['total'], dtype=bool)
            data = [[0.0, 0.0] for i in range(self.nodes['total'])]
            with open(os.path.join(self.path, name), 'r', encoding='utf-8') as f:
                for line in f:
                    th = line.split('\t')
                    node_id, node_type, node_label = int(th[0]), int(
                        th[1]), list(map(float, th[2].split(',')))
                    mask[node_id] = True
                    data[node_id] = node_label
                    labels['count'][node_type] += 1
                    labels['total'] += 1
            labels['num_labels'] = nl
            labels['data'] = np.array(data)
            labels['mask'] = mask
            
        return labels

    def get_node_type(self, node_id):
        for i in range(len(self.nodes['shift'])):
            if node_id < self.nodes['shift'][i]+self.nodes['count'][i]:
                return i

    def get_edge_type(self, info):
        if type(info) is int or len(info) == 1:
            return info
        for i in range(len(self.links['meta'])):
            if self.links['meta'][i] == info:
                return i
        info = (info[1], info[0])
        for i in range(len(self.links['meta'])):
            if self.links['meta'][i] == info:
                return -i - 1
        raise Exception('No available edge type')

    def get_edge_info(self, edge_id):
        return self.links['meta'][edge_id]

    def list_to_sp_mat(self, li):
        data = [x[2] for x in li]
        i = [x[0] for x in li]
        j = [x[1] for x in li]
        return sp.coo_matrix((data, (i, j)), shape=(self.nodes['total'], self.nodes['total'])).tocsr()