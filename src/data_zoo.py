import os
import pickle
import logging
import shutil

import numpy as np
import torch
from torch_geometric.datasets import Planetoid, Coauthor, Reddit, Flickr, Yelp
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.data import Data
import networkx as nx
from ogb.nodeproppred import PygNodePropPredDataset, NodePropPredDataset
import scipy.sparse as sp
from torch_geometric.datasets import Yelp
from torch_geometric.data import Data
import torch_geometric.transforms as T
from sklearn.decomposition import PCA
import src.config as config


class DataProcessing:
    def __init__(self, args):
        self.logger = logging.getLogger('data_zoo')
        self.args = args

        self.dataset_name = self.args['dataset_name']
        self.num_features = {
            "cora": 1433,
            "pubmed": 500,
            "citeseer": 3703,
            "CS": 6805,
            "Physics": 100, # 8415           
            "ogbn-arxiv": 128,
            "ogbn-products": 100,
            "reddit": 602,  # Reddit has 602 features
            "flickr": 500,  # Flickr has 500 features
            "yelp": 100,  # Yelp has 100 features
            "twitch": 500,  # Twitch has 500 features
            "ogbn-proteins": 100,  # OGBN-Proteins has 100 features
            "ogbn-mag": 128,  # OGBN-MAG has 128 features
            "ogbn-papers100M": 128,  # OGBN-Papers100M has 128 features
        }
        self.target_model = self.args['target_model']

        self.determine_data_path()

    def determine_data_path(self):
        embedding_name = '_'.join(('embedding', self.args['unlearn_task'], str(self.args['unlearn_ratio'])))

        target_model_name = '_'.join((self.target_model, self.args['unlearn_task'], str(self.args['unlearn_ratio'])))
        optimal_weight_name = '_'.join((self.target_model, self.args['unlearn_task'], str(self.args['unlearn_ratio'])))

        processed_data_prefix = config.PROCESSED_DATA_PATH + self.dataset_name + "/"
        self.train_test_split_file =  processed_data_prefix + "train_test_split" + str(self.args['test_ratio'])
        self.train_data_file = processed_data_prefix + "train_data"
        self.train_graph_file = processed_data_prefix + "train_graph"
        self.embedding_file = processed_data_prefix + embedding_name
        self.unlearned_file = processed_data_prefix+ '_'.join(('unlearned', self.args['unlearn_task'], str(self.args['unlearn_ratio'])))

        self.target_model_file = config.MODEL_PATH + self.dataset_name + '/' + target_model_name
        self.optimal_weight_file = config.ANALYSIS_PATH + 'optimal/' + self.dataset_name + '/' + optimal_weight_name
        self.posteriors_file = config.ANALYSIS_PATH + 'posteriors/' + self.dataset_name + '/' + target_model_name

        dir_lists = [s + self.dataset_name for s in [config.PROCESSED_DATA_PATH,
                                                     # config.MODEL_PATH,
                                                     config.ANALYSIS_PATH + 'optimal/',
                                                     config.ANALYSIS_PATH + 'posteriors/']]
        for dir in dir_lists:
            self._check_and_create_dirs(dir)

    def sample_graph_by_nodes(self, data, sample_ratio=0.1):
        num_nodes = data.x.size(0)
        num_samples = int(num_nodes * sample_ratio)
        
        # Randomly sample nodes
        perm = torch.randperm(num_nodes)[:num_samples]
        
        # Get induced subgraph
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[perm] = True
        
        # Sample nodes and corresponding features
        sampled_x = data.x[perm]
        
        # Get edges within sampled nodes (induced subgraph)
        edge_mask = mask[data.edge_index[0]] & mask[data.edge_index[1]]
        sampled_edge_index = data.edge_index[:, edge_mask]
        
        # Relabel nodes to have consecutive indices
        node_idx = torch.zeros(num_nodes, dtype=torch.long)
        node_idx[perm] = torch.arange(num_samples)
        sampled_edge_index = node_idx[sampled_edge_index]
        
        return sampled_x, sampled_edge_index

    def sample_graph_by_edges(self, data, sample_ratio=0.1):
        num_edges = data.edge_index.size(1)
        num_samples = int(num_edges * sample_ratio)
        
        # Randomly sample edges
        perm = torch.randperm(num_edges)[:num_samples]
        sampled_edge_index = data.edge_index[:, perm]
        
        # Get unique nodes in sampled edges
        unique_nodes = torch.unique(sampled_edge_index)
        
        # Get node features for sampled nodes
        sampled_x = data.x[unique_nodes]
        
        # Relabel nodes to have consecutive indices
        node_idx = torch.zeros(data.x.size(0), dtype=torch.long)
        node_idx[unique_nodes] = torch.arange(len(unique_nodes))
        sampled_edge_index = node_idx[sampled_edge_index]
        
        return sampled_x, sampled_edge_index

    def sample_largest_component(self, data, sample_ratio=0.1):
        # Convert to networkx
        G = to_networkx(data)
        
        # Get largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        
        # Sample nodes from largest component
        num_samples = int(len(largest_cc) * sample_ratio)
        sampled_nodes = list(largest_cc)[:num_samples]
        
        # Get subgraph
        subgraph = G.subgraph(sampled_nodes)
        
        # Convert back to PyG data
        sampled_data = from_networkx(subgraph)
        
        # Get corresponding node features
        node_mapping = {node: i for i, node in enumerate(sampled_nodes)}
        sampled_x = data.x[[node_mapping[n] for n in sampled_nodes]]
        
        return sampled_x, sampled_data.edge_index


    def _check_and_create_dirs(self, folder):
        if not os.path.exists(folder):
            try:
                self.logger.info("checking directory %s", folder)
                os.makedirs(folder, exist_ok=True)
                self.logger.info("new directory %s created", folder)
            except OSError as error:
                self.logger.info("deleting old and creating new empty %s", folder)
                shutil.rmtree(folder)
                os.mkdir(folder)
                self.logger.info("new empty directory %s created", folder)
        else:
            self.logger.info("folder %s exists, do not need to create again.", folder)

    def load_raw_data(self):
        self.logger.info('loading raw data')
        if not self.args['is_use_node_feature']:
            self.transform = T.Compose([
                T.OneHotDegree(-2, cat=False)
            ])
        else:
            self.transform = None

        if self.dataset_name in ["cora", "pubmed", "citeseer"]:
            dataset = Planetoid(config.RAW_DATA_PATH, self.dataset_name, transform=T.NormalizeFeatures())
            labels = np.unique(dataset.data.y.numpy())
            data = dataset[0]
            
        elif self.dataset_name in ["CS", "Physics"]:
            if self.dataset_name == "Physics":
                dataset = Coauthor(config.RAW_DATA_PATH, name="Physics", pre_transform=self.transform)
                features = dataset[0].x.cpu().numpy()
                
                n_components = 100
                pca = PCA(n_components=n_components)
                pca_result = pca.fit_transform(features)
                
                pca_result_feature = torch.tensor(pca_result, dtype=torch.float32).to(dataset[0].x.device)
                del dataset[0].x
                dataset[0].x = pca_result_feature

            else:
                dataset = Coauthor(config.RAW_DATA_PATH, name="CS", pre_transform=self.transform)
            data = dataset[0]

        elif self.dataset_name in ["ogbn-arxiv", "ogbn-products", "ogbn-mag", "ogbn-papers100M"]:
            dataset = PygNodePropPredDataset(root=config.RAW_DATA_PATH, name=self.dataset_name, transform=self.transform)
            #dataset = NodePropPredDataset(root=config.RAW_DATA_PATH, name=self.dataset_name)
            data = dataset[0]
            # The data split is implemented in LGU.train_test_split()
            # split_idx = dataset.get_idx_split()
            
            #data.train_mask = split_idx['train']
            #data.val_mask = split_idx['valid']
            #data.test_mask = split_idx['test']
            
        elif self.dataset_name == "reddit":
            dataset = Reddit(root=config.RAW_DATA_PATH)
            data = dataset[0]
        elif self.dataset_name == "flickr":
            dataset = Flickr(root=config.RAW_DATA_PATH + 'flickr/')
            data = dataset[0]
        elif self.dataset_name == "yelp":
            # dataset = Yelp(root=config.RAW_DATA_PATH + 'yelp/')
            data = self.load_yelp_raw() #dataset[0]
        else:
            raise Exception('unsupported dataset')

        data.name = self.dataset_name

        return data

    # Method 1: Using PyG's built-in Yelp dataset loader
    def load_yelp_pyg(self):
        try:
            dataset = Yelp(root='data/Yelp')
            data = dataset[0]  # Get the first graph
            print(f"Loaded Yelp dataset with:")
            print(f"Number of nodes: {data.num_nodes}")
            print(f"Number of edges: {data.num_edges}")
            print(f"Number of features: {data.num_features}")
            print(f"Number of classes: {data.num_classes}")
            return data
        except Exception as e:
            print(f"Error loading Yelp dataset: {e}")
            return None

    # Method 2: Loading from raw files

    def load_yelp_raw(self):
        try:
            base_path = '/home/zitong/graph_unlearning/IDEA/IDEA/datasets/raw_data/yelp/raw'
            
            # Load adjacency matrix using scipy.sparse.load_npz
            print("Loading adjacency matrix...")
            adj_path = os.path.join(base_path, 'adj_full.npz')
            if not os.path.exists(adj_path):
                raise FileNotFoundError(f"Adjacency matrix not found at {adj_path}")
            adj_matrix = sp.load_npz(adj_path)  # Use scipy.sparse.load_npz instead of np.load
            
            # Convert sparse matrix to edge_index format
            print("Converting to edge_index format...")
            adj_coo = adj_matrix.tocoo()
            edge_index = torch.tensor([adj_coo.row, adj_coo.col], dtype=torch.long)
            
            # Load features with allow_pickle=True
            print("Loading features...")
            feat_path = os.path.join(base_path, 'feats.npy')
            if not os.path.exists(feat_path):
                raise FileNotFoundError(f"Features not found at {feat_path}")
            features = np.load(feat_path, allow_pickle=True)
            x = torch.from_numpy(features).float()
            
            # Load labels with allow_pickle=True
            print("Loading labels...")
            label_path = os.path.join(base_path, 'labels.npy')
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Labels not found at {label_path}")
            labels = np.load(label_path, allow_pickle=True)
            y = torch.from_numpy(labels).long()
            
            # Create PyG Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                y=y
            )
            
            print(f"""
            Successfully loaded Yelp dataset:
            Nodes: {data.num_nodes}
            Edges: {data.num_edges}
            Features: {data.num_features}
            Classes: {len(torch.unique(data.y))}
            """)
            
            return data
            
        except Exception as e:
            print(f"Error loading Yelp dataset: {e}")
            print(f"Current working directory: {os.getcwd()}")
            return None


    def load_yelp_raw_v0(self):
        try:
            base_path = config.RAW_DATA_PATH + 'yelp/raw' #'/home/zitong/graph_unlearning/IDEA/IDEA/datasets/raw_data/yelp/raw'
            
            # Load adjacency matrix
            print("Loading adjacency matrix...")
            adj_matrix = sp.load_npz(os.path.join(base_path, 'adj_full.npz'))
            
            # Convert to edge_index
            adj_coo = adj_matrix.tocoo()
            edge_index = torch.tensor([adj_coo.row, adj_coo.col], dtype=torch.long)
            
            # Load features
            print("Loading features...")
            features = np.load(os.path.join(base_path, 'feats.npy'))
            x = torch.from_numpy(features).float()
            
            # Load labels
            print("Loading labels...")
            labels = np.load(os.path.join(base_path, 'labels.npy'))
            y = torch.from_numpy(labels).long()
            
            # Create PyG Data object
            data = Data(x=x, edge_index=edge_index, y=y)
            
            return data
            
        except Exception as e:
            print(f"Error loading raw files: {e}")
            return None

    # Method 3: Complete loading with splits
    def load_yelp_complete(self):
        try:
            base_path = '/home/zitong/graph_unlearning/IDEA/IDEA/datasets/raw_data/yelp/raw'
            
            # Load adjacency matrix
            print("Loading adjacency matrix...")
            adj_matrix = sp.load_npz(os.path.join(base_path, 'adj_full.npz'))
            adj_coo = adj_matrix.tocoo()
            edge_index = torch.tensor([adj_coo.row, adj_coo.col], dtype=torch.long)
            
            # Load features
            print("Loading features...")
            features = np.load(os.path.join(base_path, 'feats.npy'))
            x = torch.from_numpy(features).float()
            
            # Load labels
            print("Loading labels...")
            labels = np.load(os.path.join(base_path, 'labels.npy'))
            y = torch.from_numpy(labels).long()
            
            # Load or create train/val/test masks
            print("Loading splits...")
            num_nodes = x.size(0)
            
            # Try to load existing splits
            try:
                splits = np.load(os.path.join(base_path, 'splits.npy'), allow_pickle=True)
                train_mask = torch.from_numpy(splits['train_mask'])
                val_mask = torch.from_numpy(splits['val_mask'])
                test_mask = torch.from_numpy(splits['test_mask'])
            except:
                print("Creating random splits...")
                # Create random splits if not available
                indices = torch.randperm(num_nodes)
                train_mask = torch.zeros(num_nodes, dtype=torch.bool)
                val_mask = torch.zeros(num_nodes, dtype=torch.bool)
                test_mask = torch.zeros(num_nodes, dtype=torch.bool)
                
                # 60/20/20 split
                train_mask[indices[:int(0.6*num_nodes)]] = True
                val_mask[indices[int(0.6*num_nodes):int(0.8*num_nodes)]] = True
                test_mask[indices[int(0.8*num_nodes):]] = True
            
            # Create Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                y=y,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask
            )
            
            print(f"""
            Dataset statistics:
            Nodes: {data.num_nodes}
            Edges: {data.num_edges}
            Features: {data.num_features}
            Classes: {len(torch.unique(data.y))}
            Train samples: {train_mask.sum().item()}
            Val samples: {val_mask.sum().item()}
            Test samples: {test_mask.sum().item()}
            """)
            
            return data
            
        except Exception as e:
            print(f"Error in complete loading: {e}")
            return None
    

    def save_train_data(self, train_data):
        self.logger.info('saving train data')
        pickle.dump(train_data, open(self.train_data_file, 'wb'))

    def load_train_data(self):
        self.logger.info('loading train data')
        return pickle.load(open(self.train_data_file, 'rb'))

    def save_train_graph(self, train_data):
        self.logger.info('saving train graph')
        pickle.dump(train_data, open(self.train_graph_file, 'wb'))

    def load_train_graph(self):
        self.logger.info('loading train graph')
        return pickle.load(open(self.train_graph_file, 'rb'))

    def save_train_test_split(self, train_indices, test_indices):
        self.logger.info('saving train test split data')
        pickle.dump((train_indices, test_indices), open(self.train_test_split_file, 'wb'))

    def load_train_test_split(self):
        self.logger.info('loading train test split data')
        return pickle.load(open(self.train_test_split_file, 'rb'))

    def save_embeddings(self, embeddings):
        self.logger.info('saving embedding data')
        pickle.dump(embeddings, open(self.embedding_file, 'wb'))

    def load_embeddings(self):
        self.logger.info('loading embedding data')
        return pickle.load(open(self.embedding_file, 'rb'))

    def load_unlearned_data(self, suffix):
        file_path = '_'.join((self.unlearned_file, suffix))
        self.logger.info('loading unlearned data from %s' % file_path)
        return pickle.load(open(file_path, 'rb'))

    def save_unlearned_data(self, data, suffix):
        self.logger.info('saving unlearned data %s' % suffix)
        pickle.dump(data, open('_'.join((self.unlearned_file, suffix)), 'wb'))

    def save_target_model(self, run, model, suffix=''):
        model.save_model(self.target_model_file + '_' + str(run))

    def load_target_model(self, run, model, suffix=''):
        model.load_model(self.target_model_file + '_'  + '_' + str(0))

    def save_optimal_weight(self, weight, run):
        torch.save(weight, self.optimal_weight_file + '_' + str(run))

    def load_optimal_weight(self, run):
        return torch.load(self.optimal_weight_file + '_' + str(run))

    def save_posteriors(self, posteriors, run, suffix=''):
        torch.save(posteriors, self.posteriors_file + '_' + str(run) + suffix)

    def load_posteriors(self, run):
        return torch.load(self.posteriors_file + '_' + str(run))

    def _extract_embedding_method(self, partition_method):
        return partition_method.split('_')[0]

class GraphDataManager:
    def __init__(self, base_dir="processed_graphs"):
        self.base_dir = base_dir
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

    def save_processed_graph(self, data, name, removed_info = []):
        """
        Save the processed graph data and removal information
        
        Args:
            data: PyG Data object from Planetoid dataset
            removed_info: Dict containing information about removed nodes/edges
            name: Name identifier for the saved data
        """
        save_dict = {
            'edge_index': data.edge_index,
            'x': data.x,
            'y': data.y,
            'train_mask': data.train_mask,
            'val_mask': data.val_mask,
            'test_mask': data.test_mask,
            'removed_info': removed_info,
            'num_features': data.num_features
        }
        
        save_path = os.path.join(self.base_dir, f"{name}.pt")
        torch.save(save_dict, save_path)
        
    def load_processed_graph(self, name):
        """
        Load the processed graph data as PyG Data object
        
        Args:
            name: Name identifier of the saved data
            
        Returns:
            data: PyG Data object
            removed_info: Information about removed nodes/edges
        """
        load_path = os.path.join(self.base_dir, f"{name}.pt")
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No saved graph found at {load_path}")
            
        saved_dict = torch.load(load_path)
        
        # Create a PyG Data object with the loaded attributes
        data = Data(
            x=saved_dict['x'],
            edge_index=saved_dict['edge_index'],
            y=saved_dict['y']
        )
        
        # Add masks
        data.train_mask = saved_dict['train_mask']
        data.val_mask = saved_dict['val_mask']
        data.test_mask = saved_dict['test_mask']
        data.num_features = saved_dict['num_features']
        
        return data, saved_dict['removed_info']

# Example usage:
def process_and_save_graph(self, delete_nodes=None, delete_edges=None):
    """
    Process graph by removing nodes/edges and save the result
    """
    # Assuming self.data is already a PyG Data object
    removed_info = {
        'deleted_nodes': delete_nodes if delete_nodes is not None else [],
        'deleted_edges': delete_edges if delete_edges is not None else []
    }
    
    # Process the graph (your existing node/edge removal logic here)
    if delete_edges is not None:
        self.data.edge_index = self.update_edge_index_unlearn(delete_nodes, delete_edges)
    
    # Save processed graph
    graph_manager = GraphDataManager()
    graph_manager.save_processed_graph(
        self.data,
        removed_info,
        f"processed_graph_{self.dataset_name}"
    )

def load_processed_graph(self):
    """
    Load previously processed graph
    """
    graph_manager = GraphDataManager()
    try:
        loaded_data, removed_info = graph_manager.load_processed_graph(
            f"processed_graph_{self.dataset_name}"
        )
        return loaded_data, removed_info
    except FileNotFoundError:
        print("No processed graph found. Please process the graph first.")
        return None, None

# Usage example:
"""
# When processing and saving:
dataset = Planetoid(config.RAW_DATA_PATH, self.dataset_name, transform=T.NormalizeFeatures())
data = dataset[0]
# After modifying the graph
process_and_save_graph(delete_nodes=nodes_to_remove)

# When loading for training:
loaded_data, removed_info = load_processed_graph()
if loaded_data is not None:
    # Use loaded_data for training
    # loaded_data will be the same PyG Data class as dataset[0]
"""
