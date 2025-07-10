import logging
import os
import gc
import networkx as nx

import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

torch.cuda.empty_cache()
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborSampler, NeighborLoader, ClusterData, ClusterLoader
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from ogb.nodeproppred import Evaluator
from torch.autograd import grad
import numpy as np
from time import time

from src.gnn_zoo import GNNBase, GATNet, GINNet, GCNNet, SGCNet,GraphSAGE, GCN
from src.data_zoo import GraphDataManager
from src.parameter_parser import parameter_parser
import src.utils as utils
import src.config as config

from tqdm import tqdm

EXTRA_INFO_PATH = "/raid/zitong/scalable_GNN_unlearning/extra_info"

class GradientTracker:
    def __init__(self):
        self.grad_norms = []
        self.layer_grad_norms = {}
        self.param_grad_changes = {}
        self.prev_grads = {}
    
    def track_gradients(self, model, epoch):
        total_norm = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                
                # Track per-layer gradient norms
                if name not in self.layer_grad_norms:
                    self.layer_grad_norms[name] = []
                self.layer_grad_norms[name].append(param_norm.item())
                
                # Track gradient changes
                if name in self.prev_grads:
                    grad_change = (param.grad.data - self.prev_grads[name]).norm(2).item()
                    if name not in self.param_grad_changes:
                        self.param_grad_changes[name] = []
                    self.param_grad_changes[name].append(grad_change)
                
                # Store current gradients
                self.prev_grads[name] = param.grad.data.clone()
        
        total_norm = total_norm ** 0.5
        self.grad_norms.append(total_norm)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Total gradient norm: {total_norm:.4f}')

class NodeClassifier(GNNBase):
    def __init__(self, num_feats, num_classes, args, hidden_channels = 16, data=None):
        super(NodeClassifier, self).__init__()
        self.args = args
        self.logger = logging.getLogger('node_classifier')
        self.target_model = args['target_model']
        self.num_classes = num_classes
        
        # set cuda number based on self.args['cuda'] to self.device
        if(self.args['cuda'] > -1):
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.args['cuda'])
            self.device = torch.device('cuda:%d' % self.args['cuda'] if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        self.data = data
        self.model = self.determine_model(num_feats, hidden_channels, num_classes).to(self.device)
        self.loss_all = None
        self.attack_preparations = {"unlearned_feature_pre":[], "retrained_predicted_prob":[],"predicted_prob":[]}
        
        if(args["hessian_approximation_method"] in ["importancesampling","adaptiveselection"]):
            self.compute_sampling_weights(edge_index=self.data.edge_index, num_nodes=self.data.num_nodes, alpha=0.85)
    
    def compute_sampling_weights(self, edge_index, num_nodes, alpha=0.85):
        """
        Compute sampling weights based on different node importance metrics
        
        Args:
            edge_index: PyG edge_index tensor
            num_nodes: Number of nodes in the graph
            method: 'degree' or 'pagerank'
            alpha: Damping factor for PageRank
        
        Returns:
            sampling_weights: Normalized importance weights for each node
        """
        self.logger.info('Begin computing/loading sampling weights')
        
        if(os.path.exists(os.path.join(EXTRA_INFO_PATH, self.args['dataset_name']+'_pagerank_weights.npy'))):
            self.pagerank_weights = np.load(os.path.join(EXTRA_INFO_PATH, self.args['dataset_name']+'_pagerank_weights.npy'))
            self.degree_weights = np.load(os.path.join(EXTRA_INFO_PATH, self.args['dataset_name']+'_degree_weights.npy'))
            self.logger.info('Sampling weights loaded')
            return
        
        # Convert to NetworkX graph for easier computation
        edge_list = edge_index.t().numpy()
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(edge_list)
        
        # Get node degrees
        degrees = dict(G.degree())
        degree_weights = np.array([degrees.get(i, 0) for i in range(num_nodes)])
        
        # Compute PageRank scores
        pagerank = nx.pagerank(G, alpha=alpha)
        pagerank_weights = np.array([pagerank.get(i, 0) for i in range(num_nodes)])
        
        # Add small constant to avoid zero probabilities
        degree_weights = degree_weights + 1e-10
        pagerank_weights = pagerank_weights + 1e-10
        
        # Normalize weights to get probability distribution
        degree_weights = degree_weights / degree_weights.sum()
        pagerank_weights = pagerank_weights / degree_weights.sum()
        
        self.degree_weights = degree_weights
        self.pagerank_weights = pagerank_weights
        
        # save pagerank_weights and degree_weights to EXTRA_INFO_PATH
        if not os.path.exists(EXTRA_INFO_PATH):
            os.makedirs(EXTRA_INFO_PATH)
            
        np.save(os.path.join(EXTRA_INFO_PATH, self.args['dataset_name']+'_pagerank_weights.npy'), pagerank_weights)
        np.save(os.path.join(EXTRA_INFO_PATH, self.args['dataset_name']+'_degree_weights.npy'), degree_weights)
        
        self.logger.info('Sampling weights computed')

    def determine_model(self, num_feats, hidden_channels, num_classes):
        self.logger.info('target model: %s' % (self.args['target_model'],))

        if self.target_model == 'GAT':
            self.lr, self.decay = 0.01, 0.001
            return GATNet(num_feats, num_classes)
        elif self.target_model == 'GCN':
            self.lr, self.decay = 0.01, 0.0001
            if(self.args['dataset_name'] in ["ogbn-arxiv","Physics","ogbn-products","reddit","CS","ogbn-papers100M"]):
                return GCN(
                            in_channels=num_feats,
                            hidden_channels=64, # originally it's 256
                            out_channels=num_classes,
                            num_layers=2,
                            dropout=0.5
                        )
        elif self.target_model == 'SimpleGCN':
            self.lr, self.decay = 0.01, 0.0001
            if(self.args['dataset_name'] in ["ogbn-arxiv","Physics","ogbn-products","reddit","CS","ogbn-papers100M"]):
                return GCN(
                            in_channels=num_feats,
                            hidden_channels=512,
                            out_channels=num_classes,
                            num_layers=self.args['num_GCNlayer'],
                            dropout=0.5,
                            linear_only = True
                        )
            elif(self.args['dataset_name'] in ['cora','citeseer']):
                return GCNNet(num_feats=num_feats, num_classes=num_classes, 
                    hidden_channels = 16, linear_only = True)
        elif self.target_model == 'GCNNet':
            self.lr, self.decay = 0.01, 0.0001
            return GCNNet(num_feats=num_feats, num_classes=num_classes, 
                    hidden_channels = 16)
        elif self.target_model == 'SAGE':
            self.lr, self.decay = 0.01, 0.0001
            return GraphSAGE(
                        in_channels = num_feats,
                        hidden_channels = 256,
                        out_channels = num_classes,
                        num_layers=2 )
        elif self.target_model == 'GIN':
            self.lr, self.decay = 0.01, 0.0001
            return GINNet(num_feats, num_classes)
        elif self.target_model == 'SGC':
            self.lr, self.decay = 0.05, 0.0001
            return SGCNet(num_feats, num_classes)
        elif self.target_model == 'GraphSAGE':
            self.lr, self.decay = 0.01, 0.0001
            return GraphSAGE(num_feats, num_classes)
        else:
            raise Exception('unsupported target model')

    def strongly_convex_nll_loss(self, predictions, targets, num_classes, lambda_pred=0.1, lambda_target=0.1):
        """
        Compute strongly convex NLL loss with L2 regularization on predictions and targets.
        
        :param predictions: Model predictions (log probabilities)
        :param targets: True labels (class indices)
        :param num_classes: Number of classes in the classification task
        :param lambda_pred: Regularization strength for predictions
        :param lambda_target: Regularization strength for targets
        :return: Loss value
        """
        # Standard NLL loss
        nll_loss = F.nll_loss(predictions, targets)
        
        # Convert targets to one-hot encoding
        target_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        
        # L2 regularization on predictions
        pred_reg = lambda_pred * torch.norm(torch.exp(predictions), p=2, dim=1).mean()
        
        # L2 regularization on targets
        target_reg = lambda_pred * torch.norm(torch.exp(target_one_hot), p=2, dim=1).mean()
        
        # Combine all terms
        total_loss = nll_loss + pred_reg + target_reg
        
        return total_loss
    
    def swap_x_edge_index(self, new_edge_index, old_edge_index, new_x, old_x):
        
        # swap x and edge_index
        
        self.data.old_edge_index = old_edge_index.detach().clone()
        self.data.old_x = old_x.detach().clone()
        
        self.data.edge_index = new_edge_index.detach().clone()
        self.data.x = new_x.detach().clone()
    
    def modify_data_to_remained(self, removed_nodes = [], save_remained_data = False):
        original_edge_index = self.data.edge_index.detach().clone()
        original_x = self.data.x.detach().clone()
        #original_edge_index = self.data.edge_index.clone()
        #original_x = self.data.x.clone()
        
        self.data.edge_index = self.data.edge_index_unlearn.detach().clone()
        self.data.x = self.data.x_unlearn.detach().clone()
        
        self.data.original_edge_index = original_edge_index
        self.data.original_x = original_x
        
        # remove original_edge_index and original_x
        del self.data.original_edge_index
        del self.data.original_x
        
        if(len(removed_nodes)):
            self.logger.info("number of original train_mask sum: %d" % sum(self.data.train_mask))
            self.logger.info("number of removed_nodes in modify_data_to_remained: %d" % len(removed_nodes))
            self.data.train_mask[removed_nodes] = False
            self.data.test_mask[removed_nodes] = True
        
        if(save_remained_data == True):
            assert self.args['unlearn_method'] == 'retrain'
            self.logger.info("========== saving remained data... =========")
            file_name = utils.get_remain_data_file_name(self.args['dataset_name'], self.args['unlearn_task'], self.args['unlearn_ratio'])
            graph_manager = GraphDataManager(base_dir = utils.REMAINED_GRAPH_PATH)
            graph_manager.save_processed_graph(data = self.data, name = file_name)
            self.logger.info("saving remained data done, now exit.")
            exit(1)
        
    def train_model_gif(self, unlearn_info=None, loss_func = 'scaled_sum'):
        # this code is not used in main function (large_graph_unlearning_LGU.py)
        self.logger.info("training model")
        
        self.model.reset_parameters()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.decay)
        
        if(self.args['dataset_name'] == 'ogbn-arxiv'):
            evaluator = Evaluator(name='ogbn-arxiv')
        else:
            self.model.train()
            self._gen_train_loader(_train_mask = self.data.train_mask)
            evaluator = None
        
        self.model, self.data = self.model.to(self.device), self.data.to(self.device)
        self.data.y = self.data.y.squeeze().to(self.device)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                            factor=0.5, patience=20, 
                                                            min_lr=1e-5)
        # for epoch in tqdm(range(int(self.args['num_epochs']))):
        for epoch in range(int(self.args['num_epochs'])):
            
            optimizer.zero_grad()
            
            if(self.args['dataset_name'] in ['ogbn-products', 'reddit']):
                loss, train_acc = self.train_sampler(train_loader = self.train_loader, optimizer = optimizer, device = self.device, nodes_mask = [])
                
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                    f'Train Acc: {train_acc:.4f}')
            else:
                self.train_fullbatch(train_loader = self.train_loader, optimizer = optimizer, epoch = epoch, evaluator = evaluator, scheduler = scheduler)

            """
            if self.target_model in ['GCN','SGC']:
                out = self.model.forward_once(self.data, self.edge_weight)
            else:
                out = self.model.forward_once(self.data)
            
            loss = F.nll_loss(out[self.data.train_mask], self.data.y[self.data.train_mask])
            """
            # lambda-strong convex loss function
            # conf_score, pred = torch.max(out[self.data.train_mask], dim = 1)
            # loss = F.nll_loss(out[self.data.train_mask], self.data.y[self.data.train_mask]) + (self.args['lambda']/2) * (torch.norm(pred.float() - self.data.y[self.data.train_mask], p=2))**2
            # bound the loss
            # loss = torch.clip(loss, max = self.args['c'])
            
            # loss.backward()
            
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), )
            # optimizer.step()

        grad_all, grad1, grad2 = None, None, None

        if self.target_model in ['GCN','SGC','GCNNet']:
            out1 = self.model.forward_once(self.data, self.edge_weight)
            out2 = self.model.forward_once_unlearn(self.data, self.edge_weight_unlearn) # edge_weight_unlearn the remained adj matrix
        else:
            out1 = self.model.forward_once(self.data)
            out2 = self.model.forward_once_unlearn(self.data)
        
        # when training the original model, it saves some intermediate results.
        # self.deleted_nodes, self.feature_nodes, self.influence_nodes
        if self.args["unlearn_task"] == "edge":
            mask1 = np.array([False] * out1.shape[0])
            mask1[unlearn_info[2]] = True # Ladd
            mask2 = mask1 # Lsub?
        if self.args["unlearn_task"] == "node":
            mask1 = np.array([False] * out1.shape[0]) 
            mask1[unlearn_info[0]] = True # removed nodes
            mask1[unlearn_info[2]] = True # influenced nodes
            mask2 = np.array([False] * out2.shape[0])
            mask2[unlearn_info[2]] = True # influenced nodes
        if self.args["unlearn_task"] == "feature" or self.args["unlearn_task"] == 'partial_feature':
            mask1 = np.array([False] * out1.shape[0])
            mask1[unlearn_info[1]] = True #  Ladd
            mask1[unlearn_info[2]] = True 
            mask2 = mask1 # Lsub

        if(loss_func == 'scaled_sum'):
            self.logger.info("using scaled_sum loss function")
            loss = F.nll_loss(out1[self.data.train_mask], self.data.y[self.data.train_mask], reduction='sum')/self.args['train_alpha']
            loss1 = F.nll_loss(out1[mask1], self.data.y[mask1], reduction='sum')/self.args['train_alpha']
            loss2 = F.nll_loss(out2[mask2], self.data.y[mask2], reduction='sum')/self.args['train_alpha']
        elif(loss_func == 'sum'):
            self.logger.info("using sum loss function")
            loss = F.nll_loss(out1[self.data.train_mask], self.data.y[self.data.train_mask], reduction='sum')
            loss1 = F.nll_loss(out1[mask1], self.data.y[mask1], reduction='sum')
            loss2 = F.nll_loss(out2[mask2], self.data.y[mask2], reduction='sum')
        model_params = [p for p in self.model.parameters() if p.requires_grad]
        
        
        grad_all = grad(loss, model_params, retain_graph=True, create_graph=True)
        grad1 = grad(loss1, model_params, retain_graph=True, create_graph=True) # Ladd
        grad2 = grad(loss2, model_params, retain_graph=True, create_graph=True) # Lsub

        self.loss_all = loss

        return (grad_all, grad1, grad2)

    def test_model(self, evaluator = None):
        self.model.eval()
        
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index) #, self.edge_weight)
            
        y_pred = out.argmax(dim=1).reshape(-1, 1)

        train_acc = evaluator.eval({
            'y_true': self.data.y.reshape(-1, 1)[self.data.train_mask],
            'y_pred': y_pred[self.data.train_mask],
        })['acc']
        valid_acc = evaluator.eval({
            'y_true': self.data.y.reshape(-1, 1)[self.data.test_mask],
            'y_pred': y_pred[self.data.test_mask],
        })['acc']
        test_acc = evaluator.eval({
            'y_true': self.data.y.reshape(-1, 1)[self.data.test_mask],
            'y_pred': y_pred[self.data.test_mask],
        })['acc']
        
        return train_acc, valid_acc, test_acc
        
    def gen_output_by_batch(self, loader = [], data_sampler = 'neighbor', cluster_mask = None, device = None, padding_numb_test = True):
        self.model.train()
        self.model = self.model.to(device)

        all_predictions = []
        t_train_output = 0
        t_train_trunc = 0
        t_test_output = 0
        t_test_trunc = 0
        
        def process_one_batch(batch, numb = False, print_pred_size = False):
            
            t1 = time()
            if(numb == False):
                batch = batch.to(device)
                out = self.model(batch.x, batch.edge_index)
            else:
                batch = batch.to(device)
                out = torch.zeros(batch.x.shape[0], self.num_classes).to(device)
            t2 = time()
            
            t3 = time()
            if(data_sampler == 'neighbor'):
                pred = out[:batch.batch_size]  # Only compute  on target nodes
                if(print_pred_size == True):
                    self.logger.info('gen_output_by_batch | pred size: %d batch_size: %d' % (len(pred), batch.batch_size))
            elif(data_sampler == 'cluster'): 
                if(cluster_mask == 'train'):
                    pred = out[batch.train_mask]
                elif(cluster_mask == 'test'):
                    pred = out[batch.test_mask]
            t4 = time()
            
            # calculate gradient of this batch and append it to all_grad
            all_predictions.append(pred)
            return t2 - t1, t4 - t3
            
        t_train_iter = 0
        # with torch.enable_grad(): # here, if we change no_grad() to enable_grad(), it will lead to OOM error at around batch 40.
        t1 = time()
        if('train' in loader):
            for i, batch in enumerate(self.train_loader):
                if(i % 100 == 0):
                    self.logger.info('gen_output_by_batch | train batch: %d batch_size: %d' % (i, batch.batch_size))
                t3 = time()
                t_train_p1, t_train_p2 = process_one_batch(batch, numb = False, print_pred_size=False)
                t4 = time()
                
                t_train_output += t_train_p1
                t_train_trunc += t_train_p2
                t_train_iter += t4 - t3
        elif('new_train' in loader):
            for i, batch in enumerate(self.new_train_loader):
                if(i % 100 == 0):
                    self.logger.info('gen_output_by_batch | train batch: %d' % i)
                t3 = time()
                t_train_p1, t_train_p2 = process_one_batch(batch, numb = False, print_pred_size=False)
                t4 = time()
                
                t_train_output += t_train_p1
                t_train_trunc += t_train_p2
                t_train_iter += t4 - t3
        t2 = time()
        
        self.logger.info("timing for train dataset pred: %d s. Output timing: %d s. Trunc timing: %d s. Total iter timing: %d s." % (t2 - t1, t_train_output, t_train_trunc, t_train_iter) )
        
        t1 = time()
        if('test' in loader):
            if(padding_numb_test == False):
                for i, batch in enumerate(self.test_loader):
                    t_test_p1, t_test_p2 = process_one_batch(batch, numb = False)
            elif(padding_numb_test == True):
                for i, batch in enumerate(self.test_loader):
                    if(i % 100 == 0):
                        self.logger.info('gen_output_by_batch | test batch: %d' % i)
                    t_test_p1, t_test_p2 = process_one_batch(batch, numb = True)
                    t_test_output += t_test_p1
                    t_test_trunc += t_test_p2
        elif('new_test' in loader):
            if(padding_numb_test == False):
                for i, batch in enumerate(self.new_test_loader):
                    t_test_p1, t_test_p2 = process_one_batch(batch, numb = False)
            elif(padding_numb_test == True):
                for i, batch in enumerate(self.new_test_loader):
                    if(i % 100 == 0):
                        self.logger.info('gen_output_by_batch | test batch: %d' % i)
                    t_test_p1, t_test_p2 = process_one_batch(batch, numb = True)
                    t_test_output += t_test_p1
                    t_test_trunc += t_test_p2
        t2 = time()
        
        self.logger.info("timing for test dataset pred: %d s. Output timing: %d s. Trunc timing: %d s." % (t2 - t1, t_test_output, t_test_trunc))
        
        all_predictions = torch.cat(all_predictions, dim=0)
        len_train = len(all_predictions)
        
        return all_predictions, len_train

    def gen_grad_by_batch(self, data = None, data_sampler = 'neighbor', optimizer = None, device = None, input_nodes = [], batch_size = 1000):

        self.model.train()
        
        if(not data):
            data = self.data
            
        if(not device):
            device = self.device
         
        if(data_sampler == 'neighbor'):
            if(len(input_nodes) > 0):
                train_loader = NeighborLoader(
                    data,
                    num_neighbors=[5,1], 
                    batch_size=batch_size, # only train for one batch
                    input_nodes=input_nodes,
                    shuffle=True,
                    num_workers=1
                )
                assert len(train_loader) == 1
            else:
                return None
        elif(data_sampler == 'cluster'):
            train_loader = self.cluster_loader
            
        itered_nodes = 0 
        
        for batch in train_loader:
            
            batch = batch.to(device)
            
            if(optimizer):
                optimizer.zero_grad()
            
            out = self.model(batch.x, batch.edge_index)
                
            if(data_sampler == 'neighbor'):
                y = batch.y[:batch.batch_size]  # Only compute loss on target nodes
                itered_nodes += len(y)
                #print(y.shape, out[:batch.batch_size].shape)
                if(len(y.shape) == 2):
                    y = y.squeeze()
                loss = F.nll_loss(out[:batch.batch_size], y)
                
            elif(data_sampler == 'cluster'):
                y = batch.y[batch.train_mask]
                loss = F.nll_loss(out[batch.train_mask], y)
                    
            assert loss >= 0
            
            if(optimizer):
                loss.backward()
                optimizer.step()

            return loss
    
    def gen_grad_by_batch_lbfgs(self, data = None, data_sampler = 'neighbor', optimizer = None, device = None, nodes_mask = [], batch_size = 1000):

        self.model.train()
        
        if(not data):
            data = self.data
            
        if(not device):
            device = self.device
         
        if(data_sampler == 'neighbor'):
            if(len(nodes_mask) > 0):
                train_loader = NeighborLoader(
                    data,
                    num_neighbors=[5,1], 
                    batch_size=batch_size, # only train for one batch
                    input_nodes=nodes_mask,
                    shuffle=True,
                    num_workers=1
                )
            else:
                train_loader = self.train_loader
        elif(data_sampler == 'cluster'):
            train_loader = self.cluster_loader

        def train_step(batch):
            def closure():
                optimizer.zero_grad()
                out = self.model(batch.x, batch.edge_index)
                if(data_sampler == 'neighbor'):
                    y = batch.y[:batch.batch_size]  # Only compute loss on target nodes
                    loss = F.nll_loss(out[:batch.batch_size], y)
                elif(data_sampler == 'cluster'):
                    y = batch.y[batch.train_mask]
                    loss = F.nll_loss(out[batch.train_mask], y)
                loss.backward()
                return loss

            # Perform optimization step
            loss = optimizer.step(closure)
            return loss.item()

        for batch in train_loader:
            
            batch = batch.to(device)
            loss = train_step(batch)

            return loss
        
    def train_sampler(self, data = None, data_sampler = 'neighbor', optimizer = None, device = None, nodes_mask = [], loss_func = 'scaled_sum'):
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_examples = 0
        
        if(not data):
            data = self.data
            
        if(not device):
            device = self.device
        
        if(data_sampler == 'neighbor'):
            if(len(nodes_mask) > 0):
                train_loader = NeighborLoader(
                    data,
                    num_neighbors=[5,1],
                    batch_size=512,
                    input_nodes=nodes_mask,
                    shuffle=True,
                    num_workers=4
                )
            else:
                train_loader = self.train_loader
        elif(data_sampler == 'cluster'):
            train_loader = self.cluster_loader
            
        itered_nodes = 0
        
        self.logger.info("using device: %s" % str(device))
        for batch in train_loader:
            batch = batch.to(device)
            
            if(optimizer):
                optimizer.zero_grad()
            
            if(self.target_model == 'SGC'):
                out = self.model(batch.x, batch.edge_index)
            else:
                out = self.model(batch.x, batch.edge_index)
            
            if(data_sampler == 'neighbor'):
                y = batch.y[:batch.batch_size]  # Only compute loss on target nodes
                itered_nodes += len(y)
                loss = F.nll_loss(out[:batch.batch_size], y) 
            elif(data_sampler == 'cluster'):
                y = batch.y[batch.train_mask]
                loss = F.nll_loss(out[batch.train_mask], y)
                
            assert loss >= 0
            
            if(optimizer):
                loss.backward()
                optimizer.step()
            
            if(data_sampler == 'neighbor'):
                total_loss += float(loss) * batch.batch_size
                total_correct += int((out[:batch.batch_size].argmax(dim=-1) == y).sum())
                total_examples += batch.batch_size
            elif(data_sampler == 'cluster'):
                # modify this based on batch.train_mask
                total_loss += float(loss) * batch.train_mask.sum()
                total_correct += int((out[batch.train_mask].argmax(dim=-1) == y).sum())
                total_examples += batch.train_mask.sum()
        
        self.logger.info('total itered nodes in train_sampler: %d' % itered_nodes)
        return total_loss / total_examples, total_correct / total_examples

    @torch.no_grad()
    def test_sampler(self,test_loader, device):
        self.model.eval()
        total_correct = 0
        total_examples = 0
        
        for batch in test_loader:
            batch = batch.to(device)
            out = self.model(batch.x, batch.edge_index)
            pred = out[:batch.batch_size].argmax(dim=-1)
            y = batch.y[:batch.batch_size]
            
            total_correct += int((pred == y).sum())
            total_examples += batch.batch_size
        
        return total_correct / total_examples
    
    def train_fullbatch(self, optimizer, epoch, evaluator = None, scheduler = None):
        
        self.model.train()
        optimizer.zero_grad()
            
        if self.target_model in ['GCN','GCNNet']:
            # out = self.model.forward_once(self.data, self.edge_weight)
            if(self.args['dataset_name'] == 'ogbn-arxiv'):
                out = self.model(self.data.x, self.data.edge_index)
            else:
                out = self.model.forward_once(self.data)
        elif self.target_model == 'SGC':
            out = self.model.forward_once(self.data.x, self.data.edge_index, self.edge_weight)
        else:
            out = self.model.forward_once(self.data)
            
        if(self.args['dataset_name'] == 'ogbn-arxiv'):
            out = out.log_softmax(dim=-1)                
            loss = F.nll_loss(out[self.data.train_mask], self.data.y[self.data.train_mask])
            #loss = F.cross_entropy(out[self.data.train_mask], self.data.y[self.data.train_mask])
        else:
            loss = F.nll_loss(out[self.data.train_mask], self.data.y[self.data.train_mask])
            
        loss.backward()
        
        optimizer.step()
            
        if(epoch % 10 == 0 and self.args['dataset_name'] == 'ogbn-arxiv'):
            train_acc, valid_acc, test_acc = self.test_model(evaluator, data_sampler = "none")
            self.logger.info("epoch: %s, loss: %.4f, train_acc: %.4f, valid_acc: %.4f, test_acc: %.4f" % (epoch, loss, train_acc, valid_acc, test_acc))
            scheduler.step(valid_acc)
    
    def gen_mask_for_remained_dataset(self, unlearn_info):
             
        if self.args["unlearn_task"] == "edge":
            mask3 = np.array([True] * self.data.y.shape[0])
        if self.args["unlearn_task"] == "node":
            mask3 = np.array([True] * self.data.y.shape[0])
            mask3[unlearn_info[0]] = False # remained nodes
        if self.args["unlearn_task"] == "feature" or self.args["unlearn_task"] == 'partial_feature':
            mask3 = np.array([True] *self.data.y.shape[0])
        
        mask3[self.data.test_mask] = False
        
        return mask3
    
    
    def gen_sampling_indices_for_remained_dataset(self, unlearn_info, sampling_size = 100, random_seed = 0, sampling_method = "random"):
        # sampling_method choise: {"random", "degree","pagerank"}
        
        mask3 = self.gen_mask_for_remained_dataset(unlearn_info)
            
        # Create a new array of False values with same shape
        # new_array = np.zeros_like(mask3, dtype=bool)
        
        # Get indices where True values exist
        true_indices = np.where(mask3)[0]
        
        if(sampling_method == "random"):
            # Set seed for reproducibility
            np.random.seed(random_seed)
            if len(true_indices) > 0:
                # Randomly select one index from true positions
                selected_idx = np.random.choice(true_indices, size=sampling_size, replace=False)
            else:
                # If no True values exist, randomly select from all positions
                selected_idx = np.random.choice(len(mask3), size=sampling_size, replace=False)
        elif(sampling_method == "degree"):
            if len(true_indices) > 0:
                #use self.degree_weights for importance sampling
                degree_weights = self.degree_weights[true_indices]
                degree_weights = degree_weights / np.sum(degree_weights)
                # Randomly select one index from true positions
                selected_idx = np.random.choice(true_indices, size=sampling_size, replace=False, p=degree_weights)
            else:
                selected_idx = np.random.choice(len(mask3), size=sampling_size, replace=False, p = self.degree_weights)
        elif(sampling_method == "pagerank"):
            if len(true_indices) > 0:
                #use self.degree_weights for importance sampling
                pagerank_weights = self.pagerank_weights[true_indices]
                #normalize pagerank_weights
                pagerank_weights = pagerank_weights / np.sum(pagerank_weights)
                # Randomly select one index from true positions
                selected_idx = np.random.choice(true_indices, size=sampling_size, replace=False, p=pagerank_weights)
            else:
                selected_idx = np.random.choice(len(mask3), size=sampling_size, replace=False, p = self.pagerank_weights)
        else:
            raise NotImplementedError
        # Set the selected position to True
        # new_array[selected_idx] = True
        # assert sum(new_array) == sampling_size
        
        return selected_idx
        
    def train_model(self, data_sampler = 'none', batch_size = 512):
        self.logger.info("training model remained with data sampler type %s" % data_sampler)
        
        self.model.reset_parameters()
        torch.cuda.empty_cache()
        
        self.model = self.model.to(self.device)
        self.data.y = self.data.y.squeeze().to(self.device)
        
        if(data_sampler == 'none'):
            self.data = self.data.to(self.device)
            
        self.model.train()
        self._gen_train_loader( _train_mask= self.data.train_mask, data_sampler = data_sampler, batch_size=batch_size)
        self._gen_test_loader( test_mask = self.data.test_mask, data_sampler=data_sampler, batch_size=batch_size)
        
        if self.target_model in ['GCN','SGC','GCNNet']:    
            if(self.args['load_remained_data_for_retraining'] == False):   
                self.edge_weight_unlearn = self.edge_weight_unlearn.to(self.device)
            self.edge_weight = self.edge_weight.to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.decay)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                            factor=0.5, patience=20, 
                                                            min_lr=1e-5)
        
        if(self.args['dataset_name'] == 'ogbn-arxiv'):
            evaluator = Evaluator(name='ogbn-arxiv')
        else:
            evaluator = None
            
        # for epoch in tqdm(range(int(self.args['num_epochs']))):
        for epoch in range(int(self.args['num_epochs'])):
            
            if(data_sampler in ['cluster', 'neighbor']):
                loss, train_acc = self.train_sampler(data_sampler = data_sampler, optimizer = optimizer, device = self.device, nodes_mask = [])
                
                self.logger.info(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                    f'Train Acc: {train_acc:.4f}')
                
            elif(data_sampler in ['none']):
                self.train_fullbatch(optimizer = optimizer, epoch = epoch, evaluator = evaluator, scheduler = scheduler)

    def calculte_grad_for_unlearning(self, unlearn_info=None, remained = 0, data_sampler = 'none', unlearn_method = 'none', batch_size = 512, 
                                     sampling_dataset = 0.01, memory_trace = True, random_seed = 2, loss_func = 'scaled_sum'):
        # Create a new array of False values with same shape
        
        #sampled_train_mask = np.zeros_like(self.data.train_mask, dtype=bool)
        sampled_train_mask = utils.create_zero_mask_like(self.data.train_mask)
           
        # Get indices where True values exist
        true_indices = utils.get_true_indices(self.data.train_mask)
        # true_indices = np.where(self.data.train_mask)[0]
            
        assert len(true_indices) > 0
        # Randomly select one index from true positions

        # generate new training indices from train_mask with the proportion of sampling_dataset
        if(sampling_dataset < 1):
            
            np.random.seed(random_seed)
            
            selected_idx = np.random.choice(true_indices, size= int(np.sum(self.data.train_mask.cpu().numpy()) * sampling_dataset))
            
            # Set the selected position to True
            sampled_train_mask[selected_idx] = True
            
        elif(sampling_dataset == 1):
            sampled_train_mask = self.data.train_mask
        
        elif(sampling_dataset > 1):
            
            np.random.seed(random_seed)
            
            selected_idx = np.random.choice(true_indices, size= int(sampling_dataset))
            
            # Set the selected position to True
            sampled_train_mask[selected_idx] = True
            
        # change sample_train_mask from numpy to tensor if it is a numpy ndarray
        if(type(sampled_train_mask) == np.ndarray):
            sampled_train_mask = torch.from_numpy(sampled_train_mask)
        
        # sample_test_mask is the supplement to sampled_train_mask
        sampled_test_mask = ~sampled_train_mask
        
        # self._gen_train_loader( _train_mask= self.data.train_mask, data_sampler = data_sampler, batch_size=batch_size)
        # self._gen_test_loader( test_mask = self.data.test_mask, data_sampler=data_sampler, batch_size=batch_size)
        
        self._gen_train_loader(_train_mask= sampled_train_mask, data_sampler = data_sampler, batch_size=batch_size)
        self._gen_test_loader( test_mask = sampled_test_mask, data_sampler=data_sampler, batch_size=batch_size)
            
        # when training the original model, it saves some intermediate results.
        # self.deleted_nodes, self.feature_nodes, self.influence_nodes
        if self.args["unlearn_task"] == "edge":
            mask1 = np.array([False] * self.data.y.shape[0])
            mask1[unlearn_info[2]] = True # Ladd
            # mask1 = mask1 & sampled_train_mask.numpy()
            mask1 = mask1 & utils.convert_tensor_to_numpy(sampled_train_mask)
            
            mask2 = mask1 # Lsub?
            mask3 = np.array([True] * self.data.y.shape[0])
        if self.args["unlearn_task"] == "node":
            mask1 = np.array([False] * self.data.y.shape[0])
            
            mask1[unlearn_info[0]] = True # removed nodes
            mask1[unlearn_info[2]] = True # influenced nodes
            #intersect mask1 and sampled_train_mask
            mask1 = mask1 & utils.convert_tensor_to_numpy(sampled_train_mask)
            # mask1 = mask1 & sampled_train_mask.numpy()
            
            mask2 = np.array([False] * self.data.y.shape[0])
            mask2[unlearn_info[2]] = True # influenced nodes
            # mask2 = mask2 & sampled_train_mask.numpy()
            mask2 = mask2 & utils.convert_tensor_to_numpy(sampled_train_mask)
            
            mask3 = np.array([True] * self.data.y.shape[0])
            mask3[unlearn_info[0]] = False # remained nodes
            
        if self.args["unlearn_task"] == "feature" or self.args["unlearn_task"] == 'partial_feature':
            mask1 = np.array([False] * self.data.y.shape[0])
            mask1[unlearn_info[1]] = True #  Ladd
            mask1[unlearn_info[2]] = True
            #mask1 = mask1 & sampled_train_mask.numpy()
            mask1 = mask1 & utils.convert_tensor_to_numpy(sampled_train_mask)
            
            mask2 = mask1 # Lsub
            
            mask3 = np.array([True] *self.data.y.shape[0])
            
        grad_all, grad1, grad2 = None, None, None

        if self.target_model in ['GCN','GAT', 'GIN']:

            if(data_sampler in ['cluster', 'neighbor']):
                # So this is deductive learning as we used test information during inference.
                out1, len_train_1 = self.gen_output_by_batch(loader = ['train', 'test'], data_sampler=data_sampler, device = self.device, padding_numb_test = True) # originally the loader = ['train', 'test']
                
                assert out1.shape[0] == self.data.y.shape[0]
                
                self.swap_x_edge_index(self.data.edge_index_unlearn, self.data.edge_index, self.data.x_unlearn, self.data.x)
                # now self.data.edge_index is the old edge_index_unlearn and so is the seld.data.x.
                torch.cuda.empty_cache()
                self._gen_train_loader(_train_mask= sampled_train_mask, data_sampler=data_sampler, batch_size=batch_size, new_var = True)
                # self._gen_test_loader(data_sampler=data_sampler, batch_size=batch_size, new_var = True)
                
                out_train, len_train_2 = self.gen_output_by_batch(loader = ['new_train'], data_sampler=data_sampler, device = self.device, padding_numb_test = True) # change to proxy model
                # assert len_train_1 == len_train_2
                # concat a new zero tensor shaped (out1.shape[0] - len_train_2, self.num_classes)
                # Create zero tensor with shape (out1.shape[0] - len_train_2, self.num_classes)
                zeros = torch.zeros(out1.shape[0] - out_train.shape[0], self.num_classes, device=out_train.device)
                # Concatenate along dimension 0
                out2 = torch.cat([out_train, zeros], dim=0)
                assert out2.shape[0] == self.data.y.shape[0]
                self.swap_x_edge_index(new_edge_index=self.data.old_edge_index, old_edge_index=self.data.edge_index, 
                                       new_x = self.data.old_x, old_x = self.data.x)
            elif(data_sampler in ['none']):
                if(self.args['dataset_name'] == 'ogbn-arxiv' or (self.args['dataset_name'] == 'ogbn-products' and self.target_model == 'SAGE')):
                    out1 = self.model(self.data.x, self.data.edge_index).log_softmax(dim=-1)  
                    out2 = self.model(self.data.x, self.data.edge_index_unlearn).log_softmax(dim=-1)
                else:
                    out1 = self.model.forward_once(self.data)
                    out2 = self.model.forward_once_unlearn(self.data) # edge_weight_unlearn: the remained adj matrix
        else:
            if self.target_model == 'SGC':
                out1 = self.model.forward_once(self.data.x, self.data.edge_index, self.edge_weight)
                out2 = self.model.forward_once_unlearn(self.data.x, self.data.edge_index_unlearn, self.edge_weight_unlearn)
            else:
                out1 = self.model.forward_once(self.data)
                out2 = self.model.forward_once_unlearn(self.data)
        
        # move self.data.y to device if it is not as the same device as out1
        if(out1.device != self.data.y.device):
            self.data.y = self.data.y.to(out1.device)
        
        get_grad_y = self.data.y
        
        if(len(get_grad_y) > 1):
            get_grad_y = get_grad_y.squeeze()
        
        if(loss_func == 'scaled_sum'):
            self.logger.info("using scaled_sum loss function")
            loss = F.nll_loss(out1[self.data.train_mask], get_grad_y[self.data.train_mask], reduction='sum')/self.args['train_alpha']
            loss1 = F.nll_loss(out1[mask1], get_grad_y[mask1], reduction='sum')/self.args['train_alpha']
            loss2 = F.nll_loss(out2[mask2], get_grad_y[mask2], reduction='sum')/self.args['train_alpha']
            loss_remained = F.nll_loss(out2[mask3], get_grad_y[mask3], reduction='sum')/self.args['train_alpha'] # lzt
        elif(loss_func == 'sum'):
            self.logger.info("using sum loss function")
            loss = F.nll_loss(out1[self.data.train_mask], get_grad_y[self.data.train_mask], reduction='sum')
            loss1 = F.nll_loss(out1[mask1], get_grad_y[mask1], reduction='sum')
            loss2 = F.nll_loss(out2[mask2], get_grad_y[mask2], reduction='sum')
            loss_remained = F.nll_loss(out2[mask3], get_grad_y[mask3], reduction='sum')
        
        model_params = [p for p in self.model.parameters() if p.requires_grad]
        
        if(memory_trace == True):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            initial_memory = torch.cuda.memory_allocated()
        
        if(memory_trace == True):
            # First gradient calculation
            mem_checkpoint1 = torch.cuda.memory_allocated()
        grad_all = grad(loss, model_params, retain_graph=True, create_graph=True)
        
        if(memory_trace == True):
            mem_after_grad1 = torch.cuda.memory_allocated()
        
        grad1 = grad(loss1, model_params, retain_graph=True, create_graph=False) # Ladd
        if(memory_trace == True):
            mem_after_grad2 = torch.cuda.memory_allocated()
        
        grad2 = grad(loss2, model_params, retain_graph=True, create_graph=False) # Lsub
        if(memory_trace == True):
            mem_after_grad3 = torch.cuda.memory_allocated()
        
        grad_remained = -1
        if(remained == 1):
            grad_remained = grad(loss_remained, model_params, retain_graph=True, create_graph=False) # grad(L(Dr, w0))

        self.loss_all = loss
        
        if(memory_trace == True):
            self.logger.info(f"Memory for grad_all: {(mem_after_grad1 - mem_checkpoint1) / 1024**2:.2f} MB")
            self.logger.info(f"Memory for grad1: {(mem_after_grad2 - mem_after_grad1) / 1024**2:.2f} MB")
            self.logger.info(f"Memory for grad2: {(mem_after_grad3 - mem_after_grad2) / 1024**2:.2f} MB")
            self.logger.info(f"Total memory used: {(mem_after_grad3 - initial_memory) / 1024**2:.2f} MB")
            self.memory_trace = {"grad_all":(mem_after_grad1 - mem_checkpoint1) / 1024**2, "grad1": (mem_after_grad2 - mem_after_grad1) / 1024**2,
                                 "grad2":(mem_after_grad3 - mem_after_grad2) / 1024**2 }
        else:
            self.memory_trace = {"grad_all":-1, "grad1": -1, "grad2":-1}

        return (grad_all, grad1, grad2, grad_remained)

    def train_model_continue(self, unlearn_info=None):
        self.logger.info("training model continue")
        self.model.train()
        self._gen_train_loader(_train_mask = self.data.train_mask)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=(self.lr / 1e2), weight_decay=self.decay) 
        training_mask = self.data.train_mask.clone()
        
        if unlearn_info[0] is not np.array([]):
            training_mask[unlearn_info[0]] = False

        # for epoch in tqdm(range(int(self.args['num_epochs'] * 0.1))):
        for epoch in range(int(self.args['num_epochs'] * 0.1)):
            
            optimizer.zero_grad()
            if self.target_model in ['GCN','SGC','GCNNet']:
                out = self.model.forward_once_unlearn(self.data, self.edge_weight_unlearn)

            else:
                out = self.model.forward_once_unlearn(self.data)

            loss = F.nll_loss(out[training_mask], self.data.y[training_mask])
            loss.backward()
            optimizer.step()

    def retrain_model(self, unlearn_info=None):
        
        self.logger.info("retraining model from scratch")
        
        self.model.reset_parameters()
        
        if(self.args['dataset_name'] not in ['ogbn-arxiv']):
            self.model.train()
            self._gen_train_loader( _train_mask= self.data.train_mask, batch_size=512)
        
        self.model, self.data = self.model.to(self.device), self.data.to(self.device)
        self.data.y = self.data.y.squeeze().to(self.device)
        
        if self.target_model in ['GCN','SGC','GCNNet']:        
            self.edge_weight_unlearn = self.edge_weight_unlearn.to(self.device)
            self.edge_weight = self.edge_weight.to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.decay)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                            factor=0.5, patience=20, 
                                                            min_lr=1e-5)
        
        if(self.args['dataset_name'] == 'ogbn-arxiv'):
            evaluator = Evaluator(name='ogbn-arxiv')
        else:
            evaluator = None
        
        # when training the original model, it saves some intermediate results.
        # self.deleted_nodes, self.feature_nodes, self.influence_nodes
        if self.args["unlearn_task"] == "edge":
            mask1 = np.array([False] * self.data.y.shape[0])
            mask1[unlearn_info[2]] = True # Ladd
            mask2 = mask1 # Lsub?
            mask3 = np.array([True] * self.data.y.shape[0])
        if self.args["unlearn_task"] == "node":
            mask1 = np.array([False] * self.data.y.shape[0])
            mask1[unlearn_info[0]] = True # removed nodes
            mask1[unlearn_info[2]] = True # influenced nodes
            mask2 = np.array([False] * self.data.y.shape[0])
            mask2[unlearn_info[2]] = True # influenced nodes
            
            mask3 = np.array([True] * self.data.y.shape[0])
            mask3[unlearn_info[0]] = False # remained nodes
            
        if self.args["unlearn_task"] == "feature" or self.args["unlearn_task"] == 'partial_feature':
            mask1 = np.array([False] * self.data.y.shape[0])
            mask1[unlearn_info[1]] = True #  Ladd
            mask1[unlearn_info[2]] = True 
            mask2 = mask1 # Lsub
            
            mask3 = np.array([True] * self.data.y.shape[0])
        
        # generate the and (&) operation between mask3 and self.data.train_mask
        remained_training_mask = mask3 & self.data.train_mask.cpu().numpy()
        self._gen_train_loader(_train_mask=remained_training_mask)
        
        # convert  remained_training_mask to tensor
        remained_training_mask = torch.tensor(remained_training_mask).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.decay)

        # for epoch in tqdm(range(int(self.args['num_epochs']))):
        for epoch in range(int(self.args['num_epochs'])):
            
            optimizer.zero_grad()
            
            if(self.args['dataset_name'] in ['ogbn-products', 'reddit']):
                loss, train_acc = self.train_sampler(train_loader = self.train_loader, optimizer = optimizer, device = self.device, nodes_mask = [])
                
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                    f'Train Acc: {train_acc:.4f}')
            else:
                self.train_fullbatch(train_loader = self.train_loader, optimizer = optimizer, epoch = epoch, evaluator = evaluator, scheduler = scheduler)
            
            """    
            optimizer.zero_grad()
            if self.target_model in ['GCN','SGC']:
                out = self.model.forward_once_unlearn(self.data, self.edge_weight_unlearn)
                # out = self.model.forward_once_unlearn(self.data)
            else:
                out = self.model.forward_once_unlearn(self.data)
            
            loss = F.nll_loss(out[remained_training_mask], self.data.y[remained_training_mask])
            
            loss.backward()
            
            optimizer.step()
            """
    
    def evaluate_unlearn_F1(self, new_parameters=None, data_sampler = "none", attack_prep = True): 

        if new_parameters is not None:
            idx = 0
            for p in self.model.parameters():
                p.data = new_parameters[idx]
                idx = idx + 1
        
        if(data_sampler in ["neighbor", "cluster"]):
            if(self.args["dataset_name"] == "ogbn-products"):
                test_f1 = self.evaluate_model_with_sampler(data_sampler = data_sampler, attack_prep = attack_prep)
            else:
                test_f1 = self.evaluate_model_with_sampler(data_sampler = data_sampler, attack_prep = attack_prep)
        else:
            self.model.eval()
            if self.target_model in ['GCN','GCNNet']:
                out = self.model.forward_once_unlearn(self.data, self.edge_weight_unlearn)
                # out = self.model.forward_once_unlearn(self.data)
            elif self.target_model == 'SGC':
                out = self.model.forward_once_unlearn(self.data.x, self.data.edge_index_unlearn, self.edge_weight_unlearn)
            else:
                out = self.model.forward_once_unlearn(self.data)
                
            # predicted_prob is the probability of the predicted class
            self.attack_preparations["predicted_prob"] = F.softmax(out.detach(), dim=-1)
            
            # self.attack_preparations["predicted_prob"] = 0

            self.attack_preparations["unlearned_feature_pre"] = 0

            test_f1 = f1_score(
                self.data.y[self.data['test_mask']].cpu().numpy(), 
                out[self.data['test_mask']].argmax(axis=1).cpu().numpy(), 
                average="micro"
            )
        return test_f1

    def save_model(self, save_path):
        return super().save_model(save_path)
    
    def combine_outputs_with_padding(self, train_output, test_output, total_nodes):
        """
        Combine outputs and pad to match original dataset size
        
        Args:
            train_output: Output from train_loader
            test_output: Output from test_loader
            total_nodes: Total number of nodes in original dataset
        Returns:
            Padded combined output tensor
        """
        num_classes = train_output.shape[1]
        device = train_output.device
        
        # Create empty tensor for full dataset
        combined = torch.zeros((total_nodes, num_classes), device=device)
        
        # Fill in train and test outputs
        train_indices = self.data.train_mask.nonzero().squeeze()
        test_indices = self.data.test_mask.nonzero().squeeze()
        
        # print(len(train_indices), len(test_indices), total_nodes)
        
        combined[train_indices] = train_output
        combined[test_indices] = test_output
        
        return combined

    @torch.no_grad()
    def gen_posteriors(self, data_sampler = "none"):
        
        self.model.eval()
        self.model = self.model.to(self.device)
        
        self._gen_test_loader(test_mask= self.data.test_mask, data_sampler = data_sampler)
        self._gen_train_loader( _train_mask = self.data.train_mask, data_sampler = data_sampler)
            
        y_pred_all = []
        y_prob_all = []
        
        with torch.no_grad():
            train_prediction, len_train = self.gen_output_by_batch(loader = ['train'], data_sampler='neighbor', device = self.device, padding_numb_test=False)
            test_prediction, len_test = self.gen_output_by_batch(loader = ['test'], data_sampler='neighbor', device = self.device, padding_numb_test=False)
        
        # calculate F1 based on y_pred_all and y_label_all
        
        #print(train_prediction.shape, test_prediction.shape)
        y_prob_all = self.combine_outputs_with_padding(train_prediction, test_prediction, train_prediction.shape[0] + test_prediction.shape[0])
        self.attack_preparations["predicted_prob"] = F.softmax(y_prob_all.detach(), dim=-1)
        self.attack_preparations["unlearned_feature_pre"] = 0
            
        return #test_f1

    def setup_unlearned_nodes_loader(self, unlearned_nodes, batch_size=512, num_neighbors=[10, 25]):
        """
        Set up data loader for unlearned nodes using NeighborSampler
        
        Args:
            data: PyG data object
            unlearned_nodes: Indices of unlearned nodes
            batch_size: Batch size for the loader
            num_neighbors: Number of neighbors to sample for each layer [layer1, layer2, layer3]
        """
        try:
            # Convert unlearned_nodes to tensor if needed
            if not torch.is_tensor(unlearned_nodes):
                unlearned_nodes = torch.tensor(unlearned_nodes, dtype=torch.long)
            
            # Move to CPU for sampling
            # edge_index = self.data.edge_index.cpu()
            
            # Create NeighborSampler
            self.unlearned_nodes_loader = NeighborLoader(
                self.data,
                num_neighbors=num_neighbors,
                batch_size=batch_size,
                input_nodes=unlearned_nodes,
                shuffle=True,
                num_workers=4,  # Adjust based on your system
            )
                
            print(f"\nUnlearned nodes loader created:")
            print(f"Number of unlearned nodes: {len(unlearned_nodes)}")
            print(f"Batch size: {batch_size}")
            print(f"Number of neighbors per layer: {num_neighbors}")
            print(f"Number of batches: {len(self.unlearned_nodes_loader)}")
            
        except Exception as e:
            print(f"Error setting up unlearned nodes loader: {str(e)}")
            raise
    
    @torch.no_grad()
    def evaluate_model_with_sampler(self, data_sampler = "none", attack_prep = False):
        
        self.model.eval()
        self.model = self.model.to(self.device)
        
        self._gen_test_loader(test_mask= self.data.test_mask, data_sampler = data_sampler)

        if(data_sampler == 'neighbor'):
            test_loader = self.test_loader
        elif(data_sampler == 'cluster'):
            test_loader = self.cluster_loader
        elif(data_sampler == "unlearned_nodes"):
            test_loader = self.unlearned_nodes_loader
            
        y_pred_all = []
        y_label_all = []
        y_prob_all = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.edge_index)
                
                y_pred = out.argmax(dim=-1)
                
                if(data_sampler in ['neighbor', "unlearned_nodes"] ):
                    y_pred_all.append(y_pred)
                    y_prob_all.append(out)
                elif(data_sampler == 'cluster'):
                    y_pred_all.append(y_pred[batch.test_mask])
                    y_prob_all.append(out[batch.test_mask])
                
                if(data_sampler in ['neighbor', "unlearned_nodes"] ):
                    y = batch.y# [:batch.batch_size]  # Only compute loss on target nodes
                elif(data_sampler == 'cluster'):
                    y = batch.y[batch.test_mask]
                
                y_label_all.append(y)
        
        # calculate F1 based on y_pred_all and y_label_all
        y_pred_all = torch.cat(y_pred_all, dim=0)
        y_label_all = torch.cat(y_label_all, dim=0)
        test_f1 = f1_score(
            y_label_all.cpu().numpy(),
            y_pred_all.cpu().numpy(),
            average="micro"
        )
        
        if(attack_prep == True):
            self.gen_posteriors(data_sampler = data_sampler)
            #y_prob_all = torch.cat(y_prob_all, dim=0)
            #self.attack_preparations["predicted_prob"] = F.softmax(y_prob_all.detach(), dim=-1)
            #self.attack_preparations["unlearned_feature_pre"] = 0
            
        return test_f1

    def posterior(self):
        self.logger.debug("generating posteriors")
        self.model, self.data = self.model.to(self.device), self.data.to(self.device)
        self.model.eval()

        self._gen_test_loader(test_mask=self.data.test_mask)
        if self.target_model in ['GCN','SGC',"GCNNet"]:
            if(self.args["dataset_name"] in ['ogbn-arxiv']):
                posteriors = self.model(self.data.x, self.data.edge_index)
            else:
                posteriors = self.model.inference(self.data.x, self.test_loader, self.edge_weight, self.device)
        elif(self.target_model == "SimpleGCN" and self.args["dataset_name"] in ["cora","citeseer"]):
            posteriors = self.model.inference(self.data.x, self.test_loader, self.edge_weight, self.device)
        else:
            posteriors = self.model.inference(self.data.x, self.test_loader, self.device)

        # # # only for partial feature unlearning
        # # self._gen_test_loader()
        # # if self.target_model in ['GCN','SGC']:
        # #     posteriors_partial = self.model.inference(self.data.x_unlearn, self.test_loader, self.edge_weight, self.device)
        # # else:
        # #     posteriors_partial = self.model.inference(self.data.x_unlearn, self.test_loader, self.device)
        # # self.attack_preparations["predicted_prob"] = F.softmax(posteriors_partial.detach(), dim=-1)

        for _, mask in self.data('test_mask'):
            posteriors = F.log_softmax(posteriors[mask], dim=-1)
        
        return posteriors.detach()

    def generate_embeddings(self):
        self.model.eval()
        self.model, self.data = self.model.to(self.device), self.data.to(self.device)
        self._gen_test_loader( test_mask= self.data.test_mask)

        if self.target_model in ['GCN','SGC','GCNNet']:
            logits = self.model.inference(self.data.x, self.test_loader, self.edge_weight, self.device)
        else:
            logits = self.model.inference(self.data.x, self.test_loader, self.device)
        return logits

    def _gen_train_loader(self, _train_mask = None, data_sampler = 'none', batch_size = 512, new_var = False):
        self.logger.info("generate train loader with data sampler type %s %d samples in total " % (data_sampler, sum(_train_mask)))
        train_indices = np.nonzero(self.data.train_mask.cpu().numpy())[0]
        edge_index = utils.filter_edge_index(self.data.edge_index, train_indices, reindex=False)
        
        if edge_index.shape[1] == 0:
            edge_index = torch.tensor([[1, 2], [2, 1]])

        # set random seed for these sampler
        torch.manual_seed(12345)
        if data_sampler == 'neighbor':
            # Set up neighbor sampling
            if(new_var == False):
                self.train_loader = NeighborLoader(
                    self.data,
                    num_neighbors=[5,1],  # Number of neighbors to sample for each layer, originally it's [25,10]
                    batch_size=batch_size,
                    input_nodes=_train_mask,
                    shuffle=False,
                    num_workers=4
                )
            elif(new_var == True):
                self.new_train_loader = NeighborLoader( # give a new variance for training loader
                    self.data,
                    num_neighbors=[5,1],  # Number of neighbors to sample for each layer, originally it's [25,10]
                    batch_size=batch_size,
                    input_nodes=_train_mask,
                    shuffle=False,
                    num_workers=4
                )
        elif data_sampler == 'cluster':
            # set up cluster loader based on train mask and val mask
            cluster_data = ClusterData(
                self.data,
                num_parts=1500,  # Number of clusters
                recursive=False,
                save_dir=config.CLUSTER_DATA_PATH
            )

            # Create cluster loader
            self.cluster_loader = ClusterLoader(
                cluster_data,
                batch_size=20,  # Number of clusters per batch
                shuffle=True,
                num_workers=0
            )
            
        elif data_sampler == 'none':
            self.train_loader = None
            self.cluster_loader = None

        if(self.args['load_remained_data_for_retraining'] == False):
            if self.target_model in ['GCN','SGC','GCNNet']:
                _, self.edge_weight = gcn_norm(
                    self.data.edge_index, 
                    edge_weight=None, 
                    num_nodes=self.data.x.shape[0],
                    add_self_loops=False)

                _, self.edge_weight_unlearn = gcn_norm(
                    self.data.edge_index_unlearn, 
                    edge_weight=None, 
                    num_nodes=self.data.x.shape[0],
                    add_self_loops=False)

        self.logger.info("generate train loader finish")

    def _gen_train_unlearn_load(self):
        self.logger.info("generate train unlearn loader")
        train_indices = np.nonzero(self.data.train_mask.cpu().numpy())[0]
        edge_index = utils.filter_edge_index(self.data.edge_index, train_indices, reindex=False)
        if edge_index.shape[1] == 0:
            edge_index = torch.tensor([[1, 2], [2, 1]])

        self.train_unlearn_loader = NeighborSampler(
            edge_index, node_idx=None,
            sizes=[-1], num_nodes=self.data.num_nodes,
            batch_size=self.data.num_nodes, shuffle=False,
            num_workers=0)

        if self.target_model in ['GCN','SGC','GCNNet','SimpleGCN']:
            _, self.edge_weight = gcn_norm(
                self.data.edge_index, 
                edge_weight=None, 
                num_nodes=self.data.x.shape[0],
                add_self_loops=False)

        self.logger.info("generate train loader finish")
    
    def _gen_test_loader(self, test_mask, data_sampler = "none", batch_size = 512, new_var = False):
        
        self.logger.info("generate test loader with data sampler type %s %s samples in total " % (data_sampler, sum(test_mask)))
        test_indices = np.nonzero(self.data.test_mask.cpu().numpy())[0]
        
        if not self.args['use_test_neighbors']:
            edge_index = utils.filter_edge_index(self.data.edge_index, test_indices, reindex=False)
        else:
            edge_index = self.data.edge_index

        if edge_index.shape[1] == 0:
            edge_index = torch.tensor([[1, 3], [3, 1]])
        
        if(self.args['dataset_name'] == 'ogbn-papers100M'):
            test_mask = utils.randomly_flip_ones_to_zeros(test_mask, 100000)
            
        if(data_sampler == "neighbor"):
            if(new_var == False):
                
                self.test_loader = NeighborLoader(
                    self.data,
                    num_neighbors=[5,1],  # Number of neighbors to sample for each layer
                    batch_size=batch_size,
                    input_nodes=test_mask,
                    shuffle=False,
                    num_workers=4
                )
            elif(new_var == True):
                self.new_test_loader = NeighborLoader(
                    self.data,
                    num_neighbors=[5,1],  # Number of neighbors to sample for each layer
                    batch_size=batch_size,
                    input_nodes=test_mask,
                    shuffle=False,
                    num_workers=4
                )
        elif(data_sampler == "none"):
            self.test_loader = NeighborSampler(
            edge_index, node_idx=None,
            sizes=[-1], num_nodes=self.data.num_nodes,
            batch_size=self.args['test_batch_size'], shuffle=False,
            num_workers=0)
        
        if self.target_model in ['GCN','SGC','GCNNet','SimpleGCN']:
            _, self.edge_weight = gcn_norm(self.data.edge_index, edge_weight=None, num_nodes=self.data.x.shape[0],
                                           add_self_loops=False)
            
        self.logger.info("generate test loader finish")

