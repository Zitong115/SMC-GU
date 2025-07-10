import logging
import time
import os
import torch
import copy
import src.utils
from torch.autograd import grad
from torch.autograd.functional import hessian
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd
import math

torch.cuda.empty_cache()
from torch_geometric.data import NeighborSampler
import numpy as np
from src.data_zoo import DataProcessing, GraphDataManager
from src.node_classifier import NodeClassifier
import src.config as config
import torch.nn.functional as F
import src
import csv

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'
UNLEARN_NODE_RESULT_PATH = "/home/zitong/graph_unlearning/scalable_GNN_unlearning/output/unlearn_node_F1_20250425.csv"

class Exp:
    def __init__(self, args):
        self.logger = logging.getLogger('exp')

        self.args = args
        self.data_zoo = DataProcessing(args)

    def load_data(self):
        pass

class LGU(Exp):
    def __init__(self, args):
        super(LGU, self).__init__(args)
        
        """
        if(self.args['unlearn_method'] == 'IDEA' and self.args['dataset_name'] == 'reddit'):
            if(self.args['unlearn_task'] == 'node'):
                exit(0)
            elif(self.args['unlearn_task'] == 'edge'):
                if(self.args['target_model'] == 'GCN' and self.args['unlearn_ratio'] < 0.14):
                    exit(0)
        
        if(self.args['unlearn_method'] == 'LGU' and self.args['dataset_name'] == 'ogbn-products'):
            if(self.args['unlearn_task'] == 'node' and self.args['target_model'] == 'GCN'):
                exit(0)
        """
        if(self.args['unlearn_method'] == 'IDEA'):
            if self.args['dataset_name'] == "reddit" and self.args['gif_stochastic'] == False and 'RQ5' not in self.args['file_name']:
                self.args['gif_sampling_dataset'] = 0.4
            elif(self.args['dataset_name'] == "ogbn-products" and self.args['gif_stochastic'] == False and 'RQ5' not in self.args['file_name']):
                self.args['gif_sampling_dataset'] = 0.01
        
        if(self.args['unlearn_method'] == 'IDEA' and self.args['hessian_approximation_method'] == 'hvpgif'):
            # following is the parameters used in IDEA
            if self.args['target_model'] in ["GCNNet", "SGC","GCN"]:
                if self.args['dataset_name'] in ["cora", "citeseer"]:
                    self.args['iteration'] = 10 #100 
                    self.args['scale'] = 10 #500
                elif self.args['dataset_name'] in ["CS"]:
                    self.args['gaussian_std'] = 0.1
                    self.args['iteration'] = 10 #1000
                    self.args['scale'] = 10 #5e6
                    
            elif self.args['target_model'] in ["GIN"]:
                self.args['gaussian_std'] = 0.005
                if self.args['dataset_name'] in ["cora", "citeseer"]:
                    self.args['iteration'] = 10#100
                    self.args['scale'] = 10#1000000
                elif self.args['dataset_name'] in ["CS"]:
                    self.args['iteration'] = 10#1000
                    self.args['scale'] = 10 #100000000

            elif self.args['target_model'] in ["GAT"]:
                self.args['gaussian_std'] = 0.005
                if self.args['dataset_name'] in ["cora", "citeseer"]:
                    self.args['iteration'] = 10#100
                    self.args['scale'] = 10#500000
                elif self.args['dataset_name'] in ["CS"]:
                    self.args['iteration'] = 10#1000
                    self.args['scale'] = 1 #100000000
            
            if(self.args['dataset_name'] in ["Physics", "ogbn-arxiv", "reddit", "ogbn-products"]):
                self.args['iteration'] = 1000
                self.args['scale'] = 1e4
        
        if self.args['target_model'] in ["GCN", "SGC"]:
            self.args['gaussian_std'] = 0.01
        elif self.args['target_model'] in ["GIN"]:
            self.args['gaussian_std'] = 0.005
        elif self.args['target_model'] in ["GAT"]:
            self.args['gaussian_std'] = 0.005
        
        if(self.args['dataset_name'] == 'Physics'):
            self.args['train_alpha'] = 1e3
        elif(self.args['dataset_name'] in ["ogbn-arxiv", "reddit"]):
            self.args['train_alpha'] = 1e6
        elif(self.args['dataset_name'] == "ogbn-products"):
            self.args['train_alpha'] = 1e7
        elif(self.args['dataset_name'] in ["cora","citeseer"]):
            self.args['train_alpha'] = 1e2
            
        if self.args['dataset_name'] in ["ogbn-products","reddit"]:
            if(self.args['load_remained_data_for_retraining'] == False):
                if(self.args['target_model'] == "SimpleGCN"):
                    self.args['num_epochs'] = 20
                else:
                    self.args['num_epochs'] = 20
            else:
                self.args['num_epochs'] = 20
        elif self.args['dataset_name'] in ["Physics"]:
            self.args['num_epochs'] = 20
        elif self.args['dataset_name'] in ["ogbn-arxiv"]:
            self.args['num_epochs'] = 20

        if(self.args['dataset_name'] in ["Physics"] and self.args['unlearn_task'] == 'edge'):
            self.args['l'] = 1.0
        elif(self.args['dataset_name'] in ["reddit","ogbn-arxiv"]):
            self.args['l'] = 0.05
            self.args['lambda'] = 0.5
        elif(self.args['dataset_name'] in ["ogbn-products"]):
            self.args['l'] = 0.1
            self.args['lambda'] = 0.5
        else:
            self.args['l'] = 0.1
        
        self.attack_preparations = {}

        self.logger = logging.getLogger('LGU Initiated.')
        self.deleted_nodes = np.array([])     
        self.feature_nodes = np.array([])
        self.influence_nodes = np.array([])

        self.certification_alpha1 = 0.0
        self.certification_alpha2 = 0.0
        self.samples_to_be_unlearned = 0.0

        self.originally_trained_model_params = None

        self.load_data(load_remained_data_for_retraining = self.args['load_remained_data_for_retraining'])
        self.num_feats = self.data.num_features
        self.train_test_split(load_remained_data_for_retraining = self.args['load_remained_data_for_retraining'])
        
        if(self.args['hvp_sampling_size'] == -1 and self.args['unlearn_method'] in ['IDEA', 'LGU']):
            # adjust hvp sampling size adaptively
            if(self.args['dataset_name'] in ["Physics", "reddit", "ogbn-arxiv"]):
                self.args['hvp_sampling_size'] = min(int(self.data.num_nodes / 10), 10000)
            else:
                self.args['hvp_sampling_size'] = min(int(self.data.num_nodes / 10), 5000) 
            
        if(self.args["hessian_approximation_method"] == "randomwalk"):
            src.utils.generate_walk_count(dataset_name = self.args['dataset_name'], edge_index = self.data.edge_index, num_nodes = self.data.num_nodes,
                                          output_dir = "", walk_length= self.args['random_walk_l'], batch_size = 20000, num_walk_per_node = self.args['random_walk_K'])
            self.logger.info("generate random walk count done!")
           
            base_dir = src.utils.EXTRA_INFO_PATH
            dataset_names = ["Physics" ,"ogbn-arxiv", "ogbn-products", "reddit"]  # replace with your dataset names

            # Analyze all datasets
            stats = src.utils.analyze_multiple_datasets(
                base_dir=base_dir,
                dataset_names=dataset_names,
                percentile=self.args['random_walk_selection_percentile']
            )
            
        elif(self.args["hessian_approximation_method"] == "katzindex"):
            
            if(self.args['dataset_name'] in ["ogbn-products","reddit","ogbn-arxiv"]):
                use_chunk, use_sparse, chunk_size = True, True, 512
                if(self.args['dataset_name'] == "ogbn-arxiv"):
                    max_iter = 5
                elif(self.args['dataset_name'] in ["reddit","ogbn-products"]):
                    max_iter= 1
            else:
                use_chunk, use_sparse, chunk_size = False, False, -1

            self.katz_index_matrix = src.utils.get_katz_index(dataset_name = self.args['dataset_name'], edge_index = self.data.edge_index, num_nodes = self.data.num_nodes, 
                                                              beta = 0.1, use_chunk = use_chunk, use_sparse = use_sparse, chunk_size = chunk_size, max_iter = max_iter)
            self.logger.info("calculate Katz Index done!")
            
        self.target_model_name = self.args['target_model']

        self.determine_target_model()
        
        self.unlearning_request(filter_less_important_nodes = self.args['unlearn_less_important_nodes'])
        
        run_f1 = np.empty((0))
        run_f1_unlearning = np.empty((0))
        run_f1_retrain = np.empty((0))
        run_memory_used = np.empty((0))

        unlearning_times = np.empty((0))
        training_times = np.empty((0))
        retraining_times = np.empty((0))
        my_bounds = np.empty((0))
        certified_edge_bounds = np.empty((0))
        certified_edge_worst_bounds = np.empty((0))
        actual_diffs = np.empty((0))

        self.logger.info("args: %s" % self.args)
        
        if(self.args['evaluate_model_on_unlearned_nodes'] == True):
            assert self.args["unlearn_task"] == "node"
            self.target_model.modify_data_to_remained(removed_nodes = self.attack_preparations["removed_nodes"])
            self.target_model.setup_unlearned_nodes_loader(unlearned_nodes = self.attack_preparations["removed_nodes"], batch_size=512, num_neighbors=[10, 25])
            self.logger.info("==== EVALUATE MODEL ON UNLEARNED NODES ====")
            retrain_F1, LGU_F1, IDEA_F1 = self.evaluate_model_on_unlearned_nodes()
            self.write_test_unlearn_nodes_result_to_csv(csv_file_path = UNLEARN_NODE_RESULT_PATH, retrain_F1 = retrain_F1, unlearn_F1 = LGU_F1, baseline_F1 = IDEA_F1)
            exit(0)
        
        for run in range(self.args['num_runs']):
            
            #clear memory
            torch.cuda.empty_cache()
            
            self.logger.info("======================= Run %f =========================" % run)
            
            if(self.args['unlearn_method'] not in ['retrain']):
                
                if(self.load_original_model() == False):
                    run_training_time = self._train_model(run, data_sampler = self.args['data_sampler'], modify_data_to_remained = False,
                                                        batch_size = self.args['batch_size'])
                    # if the model has not been saved, then save it to "./models"
                    if(self.args['save_original_model'] == True):
                        self.target_model.save_model(save_path = self.make_original_model_name())
                else:
                    run_training_time = -1
                    
                if(self.args["evaluation"] == True):
                    f1_score = self.evaluate(run, data_sampler = self.args['data_sampler'], attack_prep = False)
                else:
                    f1_score = -1
            else:
                f1_score, run_training_time = -1, -1
                
            run_f1 = np.append(run_f1, f1_score)
            training_times = np.append(training_times, run_training_time)
            
            self.originally_trained_model_params = [p.clone() for p in self.target_model.model.parameters() if p.requires_grad]
            self.attack_preparations["train_indices"] = self.data.train_mask
            self.attack_preparations["test_indices"] = self.data.test_mask  
            
            # unlearn and compute certification stats
            if(self.args['unlearn_method'] == 'retrain'):
                self.logger.info("==== RETRAIN MODEL ====")
                # self.target_model.modify_data_to_remained(removed_nodes = self.attack_preparations["removed_nodes"])
                if(self.args['load_remained_data_for_retraining'] == True):
                    unlearning_time = self._train_model(run, data_sampler = self.args['data_sampler'], modify_data_to_remained = False,
                                                        batch_size = self.args['batch_size'])
                else:
                    unlearning_time = self._train_model(run, data_sampler = self.args['data_sampler'], modify_data_to_remained = True,
                                                        batch_size = self.args['batch_size'], save_remained_data = self.args['save_remained_data'])
                if(self.args["evaluation"] == True):   
                    f1_score_unlearning = self.evaluate(run, data_sampler = self.args['data_sampler'], attack_prep = True)
                else:
                    f1_score_unlearning = -1
                params_change = 0.0
                memory_used = 0.0
                hessian_approximation_method = self.args['hessian_approximation_method']
            elif(self.args['unlearn_method'] == 'IDEA'):
                unlearning_time, f1_score_unlearning, params_change, memory_used, hessian_approximation_method = self.approxi(remained = self.args['use_remained_info_for_update'], hessian_approximation_method = self.args['hessian_approximation_method'],
                                                                                                attack_prep=True)
            elif(self.args['unlearn_method'] == 'LGU'):
                unlearning_time, f1_score_unlearning, params_change, memory_used, hessian_approximation_method = self.approxi(remained = -1, hessian_approximation_method = self.args['hessian_approximation_method'],
                                                                                                attack_prep=True)
            elif(self.args['unlearn_method'] == 'none'):
                unlearning_time, f1_score_unlearning, params_change, memory_used, hessian_approximation_method = 0.0, -1, 0.0, 0.0, "none"
                
            self.attack_preparations["unlearned_feature_pre"] = self.target_model.attack_preparations["unlearned_feature_pre"]
            self.attack_preparations["predicted_prob"] = self.target_model.attack_preparations["predicted_prob"]
            
            if(self.args['unlearn_method'] in ['retrain','none']):
                my_bound, certified_edge_bound, certified_edge_worst_bound, actual_diff, f1_score_retrain, retraining_time = -1, -1, -1, -1, f1_score_unlearning, 0
            elif(self.args['unlearn_method'] == "LGU"):
                my_bound, certified_edge_bound, certified_edge_worst_bound, actual_diff, f1_score_retrain, retraining_time = self.alpha_computation_LGU(params_change)
            elif(self.args['unlearn_method'] == 'IDEA'):
                my_bound, certified_edge_bound, certified_edge_worst_bound, actual_diff, f1_score_retrain, retraining_time = self.alpha_computation_gif(params_change)
            
            # save unlearned model
            if(self.args['save_unlearned_model'] == True):
                self.target_model.save_model(save_path = self.make_unlearned_model_name())
                
            self.attack_preparations["retrained_unlearned_feature_pre"] = self.target_model.attack_preparations["unlearned_feature_pre"]
            self.attack_preparations["retrained_predicted_prob"] = self.target_model.attack_preparations["predicted_prob"]

            unlearning_times = np.append(unlearning_times, unlearning_time)
            retraining_times = np.append(retraining_times, retraining_time)
            run_f1_unlearning = np.append(run_f1_unlearning, f1_score_unlearning)
            run_memory_used = np.append(run_memory_used, memory_used)

            run_f1_retrain = np.append(run_f1_retrain, f1_score_retrain)

            if(isinstance(my_bound, np.ndarray) or isinstance(my_bound, float) or isinstance(my_bound, int)):
                my_bounds = np.append(my_bounds, my_bound)
            else:
                my_bounds = np.append(my_bounds, my_bound.detach().cpu().numpy())
                
            if(isinstance(certified_edge_bound, np.ndarray) or isinstance(certified_edge_bound, float)):
                certified_edge_bounds = np.append(certified_edge_bounds, certified_edge_bound)
            else:
                certified_edge_bounds = np.append(certified_edge_bounds, certified_edge_bound.detach().cpu().numpy())
            
            certified_edge_worst_bounds = np.append(certified_edge_worst_bounds, certified_edge_worst_bound)
            actual_diffs = np.append(actual_diffs, actual_diff)

            exp_marker = [self.args['dataset_name'], self.args['unlearn_task'], str(self.args['unlearn_ratio']), self.args['target_model'], 
                          self.args['unlearn_method'], self.args['hessian_approximation_method'], str(self.args['hvp_sampling_K']),
                          str(self.args['hvp_sampling_size']), str(self.args['gif_sampling_dataset'])]
            if(self.args["unlearn_task"] == 'partial_feature'):
                exp_marker.append(str(self.args['unlearn_feature_partial_ratio']))
            exp_marker_string = "_".join(exp_marker)

            current_dir = os.getcwd()
            attack_materials_dir = os.path.join(current_dir, 'attack_materials')
            if not os.path.exists(attack_materials_dir):
                os.makedirs(attack_materials_dir)

            torch.save(self.attack_preparations, 'attack_materials/' + exp_marker_string + '.pth')
            self.logger.info("save attack materials to %s " % ('attack_materials/' + exp_marker_string + '.pth'))

        self.logger.info("Completed. \n")
        f1_score_avg = np.average(run_f1)
        f1_score_std = np.std(run_f1)

        training_time_avg = np.average(training_times)
        training_time_std = np.std(training_times)
        
        retraining_time_avg = np.average(retraining_times)
        retraining_time_std = np.std(retraining_times)

        self.logger.info("run_f1_unlearning: %s" % run_f1_unlearning)
        f1_score_unlearning_avg = np.average(run_f1_unlearning)
        f1_score_unlearning_std = np.std(run_f1_unlearning)

        memory_used_avg = np.average(run_memory_used)
        memory_used_std = np.std(run_memory_used)
        
        f1_retrain_avg = np.average(run_f1_retrain)
        f1_retrain_std = np.std(run_f1_retrain)

        unlearning_time_avg = np.average(unlearning_times)
        unlearning_time_std = np.std(unlearning_times)

        my_bounds_avg = np.average(my_bounds)
        my_bounds_std = np.std(my_bounds)

        certified_edge_bounds_avg = np.average(certified_edge_bounds)
        certified_edge_bounds_std = np.std(certified_edge_bounds)

        certified_edge_worst_bounds_avg = np.average(certified_edge_worst_bounds)
        certified_edge_worst_bounds_std = np.std(certified_edge_worst_bounds)

        actual_diffs_avg = np.average(actual_diffs)
        actual_diffs_std = np.std(actual_diffs)

        self.logger.info("f1_score: avg=%s, std=%s" % (f1_score_avg, f1_score_std))
        self.logger.info("model training time: avg=%s, std=%s seconds" % (training_time_avg, training_time_std))

        self.logger.info("retrain f1_score: avg=%s, std=%s" % (f1_retrain_avg, f1_retrain_std))

        self.logger.info("f1_score of %s: avg=%s, std=%s" % (self.args['unlearn_method'], f1_score_unlearning_avg, f1_score_unlearning_std))
        self.logger.info("unlearing time: avg=%s, std=%s " % (unlearning_time_avg, unlearning_time_std))

        self.logger.info("my_bound: avg=%s, std=%s " % (my_bounds_avg, my_bounds_std))
        self.logger.info("certified_edge_bound: avg=%s, std=%s " % (certified_edge_bounds_avg, certified_edge_bounds_std))
        self.logger.info("certified_edge_worst_bounds: avg=%s, std=%s " % (certified_edge_worst_bounds_avg, certified_edge_worst_bounds_std))
        self.logger.info("actual_diffs: avg=%s, std=%s " % (actual_diffs_avg, actual_diffs_std))

        writer = [(f1_score_avg, f1_score_std), (training_time_avg, training_time_std), (f1_score_unlearning_avg, f1_score_unlearning_std), 
                  (unlearning_time_avg, unlearning_time_std), (my_bounds_avg, my_bounds_std), (certified_edge_bounds_avg, certified_edge_bounds_std), 
                  (certified_edge_worst_bounds_avg, certified_edge_worst_bounds_std), 
                  (actual_diffs_avg, actual_diffs_std), (f1_retrain_avg, f1_retrain_std),
                  (memory_used_avg, memory_used_std), (retraining_time_avg, retraining_time_std), hessian_approximation_method]

        if self.args['write'] == True:
            self.writer_to_csv(writer)

    def evaluate_model_on_unlearned_nodes(self, retrain_model = True, unlearn_model = True, baseline_model = True):
        
        retrain_F1, unlearn_F1, baseline_F1 = -1, -1, -1
        
        if(retrain_model == True):
            # load retrained model
            model_path = self.make_unlearned_model_name(unlearn_method = "retrain", auto_adapt = True)
            self.target_model.load_model(model_path)
            retrain_F1 = self.target_model.evaluate_model_with_sampler(data_sampler = "unlearned_nodes", attack_prep = False)

        if(unlearn_model == True):
            model_path = self.make_unlearned_model_name(unlearn_method = "LGU", auto_adapt = True)
            self.target_model.load_model(model_path)
            unlearn_F1 = self.target_model.evaluate_model_with_sampler(data_sampler = "unlearned_nodes", attack_prep = False)

        if(baseline_model == True):
            model_path = self.make_unlearned_model_name(unlearn_method = "IDEA", auto_adapt = True)
            self.target_model.load_model(model_path)
            baseline_F1 = self.target_model.evaluate_model_with_sampler(data_sampler = "unlearned_nodes", attack_prep = False)

        return retrain_F1, unlearn_F1, baseline_F1
    
    def write_test_unlearn_nodes_result_to_csv(self, csv_file_path, retrain_F1, unlearn_F1, baseline_F1):
        
        if not os.path.exists(csv_file_path):
            df = pd.DataFrame(columns=["dataset", "model", "unlearn_task", "unlearn_ratio", "retrain_F1", "unlearn_F1", "baseline_F1"])
            df.to_csv(csv_file_path, index=False)

        df = pd.read_csv(csv_file_path)
        new_row = {"dataset": self.args['dataset_name'], "model": self.args['target_model'], 
                   "unlearn_task": self.args['unlearn_task'], "unlearn_ratio": self.args['unlearn_ratio'], 
                   "retrain_F1": retrain_F1, "unlearn_F1": unlearn_F1, "baseline_F1": baseline_F1 }
        
        # df = df.append(new_row, ignore_index=True)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(csv_file_path, index=False)
        
        self.logger.info("retrain_F1: %.3f, unlearn_F1: %.3f, baseline_F1: %.3f, save unlearn nodes result to %s" % (retrain_F1, unlearn_F1, baseline_F1, csv_file_path))
        
    def make_original_model_name(self):
        # make original_model_name
        original_model_name_list = [self.args['dataset_name'], self.args['target_model'], str(self.args['num_epochs']), 
                                    str(self.args['test_ratio']), 'original']
        original_model_name = "_".join(original_model_name_list)
        original_model_name = original_model_name + '.pth'
        
        original_model_path = os.path.join(config.MODEL_PATH, original_model_name)
        return original_model_path
    
    def save_original_model(self):
        
        original_model_path = self.make_original_model_name()
        
        if not os.path.exists(original_model_path):
            self.target_model.save_model(original_model_path)
            self.logger.info("original model saved to %s" % original_model_path)
    
    def load_original_model(self):
        
        original_model_path = self.make_original_model_name()

        if os.path.exists(original_model_path):
            self.logger.info("original model exists in %s" % original_model_path)
            self.target_model.load_model(original_model_path)
            return True
        else:
            self.logger.info("original model does not exist, training model..." )
            return False
        
    def load_retrain_model(self):

        # iterate all filenames in MODEL_PATH, and the result path starts with "{dataset_name}_{unlearn_task}_{unlearn_ratio}_{target_model}"
        file_name = "none"
        prefix = '_'.join([self.args['dataset_name'], self.args['unlearn_task'], str(self.args['unlearn_ratio']), self.args['target_model'], 'retrain'])
        for file in os.listdir(config.MODEL_PATH):
            if file.startswith(prefix):
                # split file name by "_" and get the 5th element
                file_name = file
                break
        
        retrain_model_path = os.path.join(config.MODEL_PATH, file_name)
        
        if os.path.exists(retrain_model_path):
            self.logger.info("retrained model exists in %s" % retrain_model_path)
            self.target_model.load_model(retrain_model_path)
            return True
        else:
            self.logger.info("retrained model does not exist, retraining model..." )
            return False
    
    def make_unlearned_model_name(self, unlearn_method = "", auto_adapt = False):
        
        hessian_approximation_method, hvp_sampling_K, hvp_sampling_size, gif_sampling_dataset = self.args['hessian_approximation_method'], str(self.args['hvp_sampling_K']), str(self.args['hvp_sampling_size']), str(self.args['gif_sampling_dataset'])
        
        def find_file_by_dataset_name_and_unlearn_method(unlearn_task, unlearn_ratio, target_model, unlearn_method):
            # search all files in config.MODEL_PATH
            for file in os.listdir(config.MODEL_PATH):
                if file.endswith(".pth") and file.startswith(self.args['dataset_name']):
                    # split file name by "_" and get the 5th element
                    file_name = file.split("_")
                    if(len(file_name) == 9):
                        if(file_name[1] == unlearn_task):
                            if(file_name[2] == str(unlearn_ratio)):
                                if(file_name[3] == target_model):
                                    if(file_name[4] == unlearn_method):
                                            return os.path.join(config.MODEL_PATH, file)
        if(auto_adapt == True):
            assert len(unlearn_method) > 0
            unlearned_model_path = find_file_by_dataset_name_and_unlearn_method(self.args['unlearn_task'], self.args['unlearn_ratio'], self.args['target_model'], unlearn_method)
            return unlearned_model_path

                
        # make unlearned_model_name
        unlearned_model_name_list = [self.args['dataset_name'], self.args['unlearn_task'], str(self.args['unlearn_ratio']), self.args['target_model'], 
                        unlearn_method if len(unlearn_method)>0 else self.args['unlearn_method'], hessian_approximation_method, hvp_sampling_K, hvp_sampling_size, gif_sampling_dataset]
        unlearned_model_name = "_".join(unlearned_model_name_list)
        unlearned_model_name = unlearned_model_name + '.pth'

        unlearned_model_path = os.path.join(config.MODEL_PATH, unlearned_model_name)
        return unlearned_model_path
    
    def load_data(self, load_remained_data_for_retraining = False):
        if(load_remained_data_for_retraining == False):
            self.data = self.data_zoo.load_raw_data()
        else:
            assert self.args['unlearn_method'] == 'retrain'
            graph_manager = GraphDataManager(base_dir = src.utils.REMAINED_GRAPH_PATH)
            self.data, removed_info = graph_manager.load_processed_graph(src.utils.get_remain_data_file_name(self.args['dataset_name'], self.args['unlearn_task'], self.args['unlearn_ratio']))
            
    def train_test_split(self, load_remained_data_for_retraining = False):
        
        if(load_remained_data_for_retraining == True):
            assert self.args['unlearn_method'] == 'retrain'
            self.logger.info('loading remained data for retraining')
            train_indices = self.data.train_mask.nonzero().squeeze()
            test_indices = self.data.test_mask.nonzero().squeeze()
            return
        
        if self.args['is_split']:
            self.logger.info('splitting train/test data')
            self.train_indices, self.test_indices = train_test_split(np.arange((self.data.num_nodes)), test_size=self.args['test_ratio'], random_state=100)
            self.data_zoo.save_train_test_split(self.train_indices, self.test_indices)

            self.data.train_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.train_indices))
            self.data.test_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.test_indices))
            self.data.val_mask = self.data.test_mask
            self.logger.info("number of training samples: %d, test:%d" % (torch.sum(self.data.train_mask), torch.sum(self.data.test_mask)))
        else:
            self.train_indices, self.test_indices = self.data_zoo.load_train_test_split()

            self.data.train_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.train_indices))
            self.data.test_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.test_indices))
    
    def unlearning_request(self, filter_less_important_nodes):
        
        if(self.args['load_remained_data_for_retraining'] == True and self.args['unlearn_method'] == 'retrain'):
            return
        
        # When update edge_index_unlearn, the sampler results should be stored.
        self.logger.debug("Total dataset  #.Nodes: %f, #.Edges: %f" % (
            self.data.num_nodes, self.data.num_edges))

        self.data.x_unlearn = self.data.x.clone()
        self.data.edge_index_unlearn = self.data.edge_index.clone()
        edge_index = self.data.edge_index.numpy()
        unique_indices = np.where(edge_index[0] < edge_index[1])[0]
        
        np.random.seed(123)
        
        if(self.args['unlearn_ratio'] > 1):
            unlearned_num = int(self.args['unlearn_ratio'])
        else:
            unlearned_num = int(len(self.train_indices) * self.args['unlearn_ratio'])
        
        if self.args["unlearn_task"] == 'node':
            unique_nodes = np.random.choice(self.train_indices,
                                            unlearned_num,
                                            replace=False)
            
            if(filter_less_important_nodes == True):
                unique_nodes = src.utils.filter_less_important_nodes(node_classifier = self.target_model, unlearning_request = unique_nodes)
            
            if(self.args['dataset_name'] == 'ogbn-papers100M'):
                self.data.edge_index_unlearn = self.data.edge_index
            else:
                self.data.edge_index_unlearn = self.update_edge_index_unlearn(unique_nodes)
                
            self.samples_to_be_unlearned = float(unlearned_num)

            self.attack_preparations["removed_nodes"] = unique_nodes

        if self.args["unlearn_task"] == 'edge':

            # setting 1
            remove_indices = np.random.choice(
                unique_indices,
                unlearned_num,
                replace=False)
            remove_edges = edge_index[:, remove_indices]
            
            if(filter_less_important_nodes == True):
                remove_edges = src.utils.filter_less_important_edges(node_classifier = self.target_model, unlearning_request = remove_edges)
            
            unique_nodes = np.unique(remove_edges)
            
            self.attack_preparations["removed_edges"] = remove_edges
            
            self.influence_nodes = copy.deepcopy(unique_nodes)
            self.data.edge_index_unlearn = self.update_edge_index_unlearn(unique_nodes, remove_indices)
            self.samples_to_be_unlearned = 0.0
            self.attack_preparations["removed_nodes"] = []
            # # setting 2
            # # sample non-existed edges
            # remove_edges = negative_sampling(self.data.edge_index, num_neg_samples=int(unique_indices.shape[0] * self.args['unlearn_ratio']))

            # # change the original graph topology to add these edges
            # self.data.edge_index = torch.cat((self.data.edge_index, remove_edges), 1)
            # self.data.edge_index_unlearn = self.data.edge_index_unlearn  # no change
            # unique_nodes = np.unique(remove_edges.numpy())

            # self.attack_preparations["removed_edges"] = remove_edges
            # self.samples_to_be_unlearned = 0.0

        if self.args["unlearn_task"] == 'feature':
            unique_nodes = np.random.choice(len(self.train_indices),
                                            unlearned_num,
                                            replace=False)
            if(filter_less_important_nodes == True):
                unique_nodes = src.utils.filter_less_important_nodes(node_classifier = self.target_model, unlearning_request = unique_nodes)
                
            self.data.x_unlearn[unique_nodes] = 0.
            self.samples_to_be_unlearned = 0.0
            self.attack_preparations["unlearned_feature_node_idx"] = unique_nodes
            self.attack_preparations["removed_nodes"] = []
        if self.args["unlearn_task"] == 'partial_feature':
            unique_nodes = np.random.choice(len(self.train_indices),
                                            unlearned_num,
                                            replace=False)
            if(filter_less_important_nodes == True):
                unique_nodes = src.utils.filter_less_important_nodes(node_classifier = self.target_model, unlearning_request = unique_nodes)
                
            unlearn_feature_dims = np.random.randint(0, self.data.x.shape[1], size=int(self.data.x.shape[1] * self.args['unlearn_feature_partial_ratio']))
            self.data.x_unlearn[unique_nodes[:, None], unlearn_feature_dims] = 0.0
            self.samples_to_be_unlearned = 0.0

            self.attack_preparations["unlearned_feature_node_idx"] = unique_nodes
            self.attack_preparations["unlearned_feature_dim_idx"] = unlearn_feature_dims
            self.attack_preparations["removed_nodes"] = []

        if(self.args["find_k_hops"]):
            self.logger.info('finding k hops')
            self.find_k_hops(unique_nodes)
            
        else:
            self.logger.info('NOT finding k hops')
            self.influence_nodes = unique_nodes
        
    def compute_hessian(self, loss):
        """Compute the full Hessian matrix"""
        # Get gradients
        grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
        grads = torch.cat([g.view(-1) for g in grads])
        
        # Initialize Hessian matrix
        n_params = len(grads)
        hessian = torch.zeros((n_params, n_params), device=grads.device)
        
        # Compute each column of the Hessian
        for i in range(n_params):
            # Compute gradient of i-th gradient component
            grad_i = torch.autograd.grad(grads[i], self.model.parameters(), retain_graph=True)
            grad_i = torch.cat([g.view(-1) for g in grad_i])
            hessian[:, i] = grad_i
            
        return hessian
    
    def stochastic_hvp(self, node_classifier, data, sampling_K=50, damping=1, sampling_method = "random"):
        """
        Compute stochastic Hessian-vector product using sampling
        
        Args:
            model: The GNN model
            data: Graph data
            sampling_K: Number of samples for stochastic estimation
            damping: Damping factor for numerical stability
        """
        node_classifier.model.eval()  # Set model to evaluation mode
        
        # Get parameters that require gradients
        params = [p for p in node_classifier.model.parameters() if p.requires_grad]
        
        # Initialize accumulator for HVP
        hvp_accum = [torch.zeros_like(p) for p in params]
        
        self.logger.info("len(params): %d. Stochastic information - sampling K: %d, sampling size: %d " % (len(params), sampling_K, self.args["hvp_sampling_size"]))
        
        for k in range(sampling_K):
            # Generate sampling mask for this iteration
            batch_indices = node_classifier.gen_sampling_indices_for_remained_dataset(unlearn_info = (self.deleted_nodes, self.feature_nodes, self.influence_nodes), 
                                                                      sampling_size = self.args["hvp_sampling_size"], random_seed = k, sampling_method = sampling_method)
            
            # Forward pass with current batch
            loss = node_classifier.gen_grad_by_batch(data=data, data_sampler='neighbor', optimizer=None, \
                                        device=None, input_nodes=batch_indices, batch_size = self.args["hvp_sampling_size"])
            
            # Calculate first-order gradients
            gradients = torch.autograd.grad(loss, params, create_graph=True)
            
            hvp_accum = self.hvp_op(params, gradients, hvp_accum, damping, sampling_K)
            """
            for param, grad, hvp_acc in zip(params, gradients, hvp_accum):
                if grad is None:
                    continue
                
                grad = -grad.flatten()
                
                # Compute Hessian-vector product for this parameter
                def hvp_function(v):
                    
                    grad_v = torch.autograd.grad(
                        grad,
                        param, 
                        v, 
                        retain_graph=True,
                        allow_unused=True
                    )[0]
                    
                    # Add damping for stability
                    if grad_v is not None:
                        return grad_v.flatten() + damping * v.flatten()
                    return damping * v.flatten()
                
                # Use conjugate gradient to solve for the HVP
                v = self.conjugate_gradient(
                    hvp_function,
                    grad, #-grad.flatten(),
                    max_iterations=20,
                    tolerance=1e-10
                )
                
                # Reshape and accumulate the result
                v = v.reshape(param.shape)
                hvp_acc.add_(v / sampling_K)  # Average across samples
            """
        return hvp_accum
    
    def hvp_op(self, params, gradients, hvp_accum, damping=1.0, sampling_K = 1):
        
        for param, grad, hvp_acc in zip(params, gradients, hvp_accum):
            if grad is None:
                continue
                
            grad = -grad.flatten()
                
            # Compute Hessian-vector product for this parameter
            def hvp_function(v):
                    
                grad_v = torch.autograd.grad(
                    grad,
                    param, 
                    v, 
                    retain_graph=True,
                    allow_unused=True
                )[0]
                    
                # Add damping for stability
                if grad_v is not None:
                    return grad_v.flatten() + damping * v.flatten()
                return damping * v.flatten()
                
            # Use conjugate gradient to solve for the HVP
            v = self.conjugate_gradient(
                hvp_function,
                grad, #-grad.flatten(),
                max_iterations=20,
                tolerance=1e-10
            )
                
            # Reshape and accumulate the result
            v = v.reshape(param.shape)
            hvp_acc.add_(v / sampling_K)  # Average across samples
        
        return hvp_accum
    
    def conjugate_gradient(self, matvec_fn, b, max_iterations=10, tolerance=1e-10):
        """
        Conjugate gradient method to solve Ax = b where A is implicit through matvec_fn
        
        Args:
            matvec_fn: Function that computes matrix-vector product
            b: The target vector
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
        """
        x = torch.zeros_like(b)
        r = b.clone()  # residual
        p = r.clone()  # search direction
        
        r_norm_sq = r.dot(r)
        
        for _ in range(max_iterations):
            Ap = matvec_fn(p)
            alpha = r_norm_sq / p.dot(Ap)
            
            x += alpha * p
            r -= alpha * Ap
            
            r_norm_sq_new = r.dot(r)
            beta = r_norm_sq_new / r_norm_sq
            r_norm_sq = r_norm_sq_new
            
            if r_norm_sq < tolerance:
                self.logger.info("Converged after %d iterations" % (_ + 1))
                break
                
            p = r + beta * p
        
        return x

    def apply_LBFGS_update(self, node_classifier, data):
        """
        Apply LBFGS update to model parameters

        Args:
            model: The GNN model
            data: Graph data
            learning_rate: Learning rate for parameter updates
        """
        node_classifier.swap_x_edge_index(node_classifier.data.edge_index_unlearn, node_classifier.data.edge_index, node_classifier.data.x_unlearn, node_classifier.data.x)

        original_model_params =  [p for p in self.target_model.model.parameters() if p.requires_grad]
        
        self.logger.info("model_params norm: %s" % (torch.norm(torch.cat([item.flatten() for item in original_model_params]), 2)))

        # Compute HVP
        params_esti = self.LBFGS_hvp(node_classifier, data) # original_model_params
        params_change = [p2 - p1 for p1, p2 in zip(original_model_params, params_esti)]

        self.logger.info("params_change norm: %s" % (torch.norm(torch.cat([item.flatten() for item in params_change]), 2)))

        node_classifier.swap_x_edge_index(new_edge_index=node_classifier.data.old_edge_index, old_edge_index=node_classifier.data.edge_index,
                                       new_x = node_classifier.data.old_x, old_x = node_classifier.data.x)

        return params_esti, params_change
    
    def LBFGS_hvp(self, node_classifier, data):
        """
        Compute HVP using LBFGS
        """
        
        lbfgs_optimizer = torch.optim.LBFGS(node_classifier.model.parameters(),
                                 lr=0.001,
                                 max_iter=20,
                                 max_eval=25,
                                 tolerance_grad=1e-7,
                                 tolerance_change=1e-9,
                                 history_size=10,
                                 line_search_fn='strong_wolfe')


        mask_remained = node_classifier.gen_mask_for_remained_dataset(unlearn_info = (self.deleted_nodes, self.feature_nodes, self.influence_nodes))
        
        loss = node_classifier.gen_grad_by_batch_lbfgs(data=data, data_sampler='neighbor', optimizer=lbfgs_optimizer, \
                                        device=None, nodes_mask = mask_remained, batch_size = int(sum(mask_remained)))
        
        node_classifier.model.eval()  # Set model to evaluation mode
        
        params = [p for p in node_classifier.model.parameters() if p.requires_grad]
        
        return params
        
    def apply_stochastic_update(self, node_classifier, learning_rate=1, sampling_K=50, sampling_method = "random"):
        """
        Apply stochastic HVP update to model parameters
        
        Args:
            model: The GNN model
            data: Graph data
            learning_rate: Learning rate for parameter updates
            sampling_K: Number of samples for stochastic estimation
        """
        
        node_classifier.swap_x_edge_index(node_classifier.data.edge_index_unlearn, node_classifier.data.edge_index, node_classifier.data.x_unlearn, node_classifier.data.x)
        
        model_params =  [p for p in self.target_model.model.parameters() if p.requires_grad]

        self.logger.info("model_params norm: %s" % (torch.norm(torch.cat([item.flatten() for item in model_params]), 2)))
        
        # Compute HVP
        if(self.args['dataset_name'] == 'Physics'):
            damping = 0.8
        elif(self.args['dataset_name'] in ['ogbn-arxiv','ogbn-products','reddit']):
            damping = 1
            
        hvp_updates = self.stochastic_hvp(node_classifier, None, sampling_K, damping, sampling_method = sampling_method)
        params_change = [h_est * learning_rate for h_est in hvp_updates]
       
        self.logger.info("learning_rate: %.5f" % learning_rate)
        self.logger.info("params_change norm: %s" % (torch.norm(torch.cat([item.flatten() for item in params_change]), 2)))
         
        params_esti   = [p1 + p2 for p1, p2 in zip(params_change, model_params)]
        
        node_classifier.swap_x_edge_index(new_edge_index=node_classifier.data.old_edge_index, old_edge_index=node_classifier.data.edge_index, 
                                       new_x = node_classifier.data.old_x, old_x = node_classifier.data.x)
        
        return params_esti, params_change

    def update_edge_index_unlearn(self, delete_nodes, delete_edge_index=None):
        edge_index = self.data.edge_index.numpy()

        unique_indices = np.where(edge_index[0] < edge_index[1])[0]
        unique_indices_not = np.where(edge_index[0] > edge_index[1])[0]
        
        if self.args["unlearn_task"] == 'edge':
            remain_indices = np.setdiff1d(unique_indices, delete_edge_index)
        else:
            unique_edge_index = edge_index[:, unique_indices]
            delete_edge_indices = np.logical_or(np.isin(unique_edge_index[0], delete_nodes),
                                                np.isin(unique_edge_index[1], delete_nodes))
            remain_indices = np.logical_not(delete_edge_indices)
            remain_indices = np.where(remain_indices == True)
       
        remain_encode = edge_index[0, remain_indices] * edge_index.shape[1] * 2 + edge_index[1, remain_indices]
        
        # unique_encode_not = edge_index[1, unique_indices_not] * edge_index.shape[1] * 2 + edge_index[0, unique_indices_not] # originally it's 2
        unique_encode_not = edge_index[1, unique_indices_not] * edge_index.shape[1] * 2 + edge_index[0, unique_indices_not]
        
        sort_indices = np.argsort(unique_encode_not)
        
        searchsorted_result = np.searchsorted(unique_encode_not, remain_encode, sorter=sort_indices).reshape(-1)
        
        len_sort_indices = len(sort_indices)
        
        # remove the element in searchsorted_result that equals len_sort_indices
        searchsorted_result = searchsorted_result[searchsorted_result < len_sort_indices]
        
        remain_indices_not = unique_indices_not[sort_indices[searchsorted_result]]
        remain_indices = np.union1d(remain_indices, remain_indices_not)
        
        return torch.from_numpy(edge_index[:, remain_indices])

    def determine_target_model(self):
        self.logger.info('target model: %s at determine_target_model' % (self.args['target_model'],))
        if(self.args['dataset_name'] == 'ogbn-papers100M'):
            num_classes = 172
            #num_classes = len(set(self.data.y.flatten().tolist()))
        else:
            num_classes = len(self.data.y.unique())
        self.target_model = NodeClassifier(num_feats = self.num_feats, num_classes = num_classes, 
                                           args = self.args, data = self.data)

    def evaluate(self, run, data_sampler = "none", attack_prep = False):
        self.logger.info('model evaluation')
        start_time = time.time()
        
        if(data_sampler in ["neighbor", "cluster"]):
            test_f1 = self.target_model.evaluate_model_with_sampler(data_sampler = data_sampler, attack_prep = attack_prep)
        else:
            posterior = self.target_model.posterior()
            
            test_f1 = f1_score(
                self.data.y[self.data['test_mask']].cpu().numpy(), 
                posterior.argmax(axis=1).cpu().numpy(), 
                average="micro"
            )

        evaluate_time = time.time() - start_time
        self.logger.info("Evaluation cost %s seconds." % evaluate_time)

        self.logger.info("Final Test F1: %s" % (test_f1,))
        return test_f1

    def _train_model(self, run, data_sampler = 'none', modify_data_to_remained = False, batch_size = 512, save_remained_data = False):
        self.logger.info('training target models, run %s' % run)

        start_time = time.time()
        
        if(modify_data_to_remained == False):
            self.target_model.data = self.data
        else:
            self.target_model.modify_data_to_remained(removed_nodes = self.attack_preparations["removed_nodes"], save_remained_data = save_remained_data)
            
        self.logger.info('number of deleted_nodes: %d' % len(self.deleted_nodes))
        self.logger.info('number of influence_nodes: %d' % len(self.influence_nodes))
        
        self.target_model.train_model(data_sampler = data_sampler, batch_size = batch_size)
            
        train_time = time.time() - start_time
        self.logger.info("Model training time: %s" % (train_time))

        return train_time
        
    def find_k_hops(self, unique_nodes):
        
        edge_index = self.data.edge_index.numpy()
        
        influenced_nodes = unique_nodes.copy()
        
        if(self.args['unlearn_method'] == "IDEA"):
            hops = 2
        else:
            hops = 1
            
        for _ in range(hops):
            target_nodes_location = np.isin(edge_index[0], influenced_nodes)
            neighbor_nodes = edge_index[1, target_nodes_location]
            influenced_nodes = np.append(influenced_nodes, neighbor_nodes)
            influenced_nodes = np.unique(influenced_nodes)
        
        neighbor_nodes = np.setdiff1d(influenced_nodes, unique_nodes)
        
        if self.args["unlearn_task"] in ['feature','partial_feature']:
            self.feature_nodes = unique_nodes
            self.influence_nodes = neighbor_nodes
        elif self.args["unlearn_task"] == 'node':
            self.deleted_nodes = unique_nodes
            self.influence_nodes = neighbor_nodes
        elif self.args["unlearn_task"] == 'edge':
            self.deleted_nodes = []
            if(self.args['unlearn_method'] == "IDEA"):
                self.influence_nodes = influenced_nodes
        
        # remove nodes that are not in train_indices from influence_nodes
        self.influence_nodes = self.influence_nodes[self.data.train_mask[self.influence_nodes]]
    
    def direct_hvp(self, node_classifier, res_tuple, remained, damping = 0.01): # originally damping = 0.9
        """
        This is for using res_tuple to calculate H-1v based on conjugate gradient.
        """
        node_classifier.model.eval()
        
        # Get parameters that require gradients
        params = [p for p in node_classifier.model.parameters() if p.requires_grad]
        
        def tuple_to_tensor(tuple_input):
            """Convert tuple of tensors into a single flattened tensor"""
            return torch.cat([t.flatten() for t in tuple_input])

        if(remained == 0):
            # make torch.tensor based on res_tuple[1] - res_tuple[2]
            v = tuple( (1/len(self.train_indices)) * (grad1 - grad2) for grad1, grad2 in zip(res_tuple[1], res_tuple[2]))
            # turn v into a tensor
            #v_tensor = tuple_to_tensor(v)
        elif(remained == 1):
            raise Exception("Not applicable in this case")
            #v_tensor = tuple_to_tensor(res_tuple[3])
        
        # grad_tensor = tuple_to_tensor(res_tuple[0])
        params_change = []
        for param, grad in zip(params, res_tuple[0]):
            if grad is None:
                continue
                
            grad = -grad.flatten() * (1/len(self.train_indices))
                
            # Compute Hessian-vector product for this parameter
            def hvp_function(v):
                    
                grad_v = torch.autograd.grad(
                    grad,
                    param, 
                    v, 
                    retain_graph=True,
                    allow_unused=True
                )[0]
                    
                # Add damping for stability
                if grad_v is not None:
                    return grad_v.flatten() + damping * v.flatten()
                return damping * v.flatten()
                
            # Use conjugate gradient to solve for the HVP
            v = self.conjugate_gradient(
                hvp_function,
                grad, #-grad.flatten(),
                max_iterations=20,
                tolerance=1e-10
            )
                
            # Reshape and accumulate the result
            v = v.reshape(param.shape)
            params_change.append(v.detach())
        
        # params_change = tuple_to_tensor(v).reshape(params.shape)
        params_esti = [p1 + p2 for p1, p2 in zip(params_change, params)]
        
        return params_esti, params_change
                
    def hvp_gif(self, remained = 0, learning_rate = -1, sampling_dataset = 1):     
        
        self.logger.info("Use hvp_gif sampling_dataset: %.5f" % sampling_dataset)
        
        res_tuple = self.target_model.calculte_grad_for_unlearning(unlearn_info=(self.deleted_nodes, self.feature_nodes, self.influence_nodes), 
                                                                   remained = remained, 
                                                                   data_sampler = self.args['data_sampler'], unlearn_method = self.args['unlearn_method'], 
                                                                   batch_size = 512, memory_trace = False, sampling_dataset=sampling_dataset,
                                                                   loss_func=self.args['loss_func']) 
        
        self.result_tuple_IDEA = res_tuple
        
        iteration, damp, scale = self.args['iteration'], self.args['damp'], self.args['scale']
        
        if(remained == 0):
            v = tuple(grad1 - grad2 for grad1, grad2 in zip(res_tuple[1], res_tuple[2]))
            h_estimate = tuple(grad1 - grad2 for grad1, grad2 in zip(res_tuple[1], res_tuple[2]))
            self.logger.info("use original IDEA approximation, len(v): %s" % len(v))
        else:
            v = tuple(- grad for grad in res_tuple[3])
            h_estimate = tuple(- grad for grad in res_tuple[3])
            
            self.logger.info("updated with remained info to approximation, len(grad_remained): %s" % len(res_tuple[3]))
        
        self.logger.info("Use hvp_gif approximation")
        
        for _ in range(iteration):

            model_params  = [p for p in self.target_model.model.parameters() if p.requires_grad]
            hv            = self.hvps(res_tuple[0], model_params, h_estimate)
            with torch.no_grad():
                h_estimate    = [ v1 + (1-damp)*h_estimate1 - hv1/scale
                            for v1, h_estimate1, hv1 in zip(v, h_estimate, hv)]

            #print the norm of h_estimate
            if( _ % 10 == 0):
                self.logger.info("iter %s,  h_estimate norm: %s" % (_, torch.norm(torch.cat([item.flatten() for item in h_estimate]), 2)))
        
        if(learning_rate > 0 and learning_rate <= 1):
            params_change = [h_est * learning_rate for h_est in h_estimate]
        else:
            params_change = [h_est / scale for h_est in h_estimate]
        params_esti   = [p1 + p2 for p1, p2 in zip(params_change, model_params)]
        
        return params_esti, params_change

    def hvp_gif_stochastic(self, remained = 0, learning_rate = -1, hvp_sampling_K = 10, use_gif_approxi = False):
        
        self.logger.info("===== Performing gif update with stochastic hvp, use_gif_approxi: %s =====" % use_gif_approxi)
        
        for i in range(hvp_sampling_K):
            
            res_tuple = self.target_model.calculte_grad_for_unlearning(unlearn_info=(self.deleted_nodes, self.feature_nodes, self.influence_nodes), 
                                                                   remained = remained, 
                                                                   data_sampler = self.args['data_sampler'], unlearn_method = self.args['unlearn_method'], 
                                                                   batch_size = 512, memory_trace = False, sampling_dataset=self.args['gif_sampling_dataset'],
                                                                   random_seed = i) 
            
            if(use_gif_approxi == True):
                params_esti, params_change = self.hvp_gif(res_tuple= res_tuple, remained = remained, learning_rate = learning_rate)
            else:
                params_esti, params_change = self.direct_hvp(node_classifier=self.target_model, res_tuple=res_tuple, remained = remained)
            if(i == 0):
                params_esti_sum = params_esti
                params_chage_sum = params_change
            else:
                params_esti_sum = [p1 + p2 for p1, p2 in zip(params_esti_sum, params_esti)]
                params_chage_sum = [p1 + p2 for p1, p2 in zip(params_chage_sum, params_change)]

        params_esti = [p / hvp_sampling_K for p in params_esti_sum]
        params_change = [p / hvp_sampling_K for p in params_change]

        return params_esti, params_change
    
    def random_walk_update(self, node_classifier, learning_rate = 1, use_extra_info = True, high_influence_nodes = None):
        
        unlearned_loss = self.target_model.gen_grad_by_batch(data=self.data, data_sampler='neighbor', optimizer=None,
                                                            device=None, input_nodes=self.deleted_nodes, batch_size = len(self.deleted_nodes))
        
        if(high_influence_nodes):
            influence_range_node = high_influence_nodes
        else:
            # load random walk counts directly.
            if self.args["unlearn_task"] == 'node':
                specific_nodes = self.deleted_nodes
            if self.args["unlearn_task"] == 'edge':
                specific_nodes = self.influenced_nodes
                
            walk_counts, metadata = src.utils.load_walk_counts_efficient(
                dataset_name = self.args['dataset_name'],
                input_dir=src.utils.EXTRA_INFO_PATH,
                nodes=specific_nodes,
                batch_size=100000
            )
            
            if(self.args['random_walk_selection_percentile'] == -1):
                if(self.args['dataset_name'] == 'ogbn-products'):
                    threshold_percentile=95
                else:
                    threshold_percentile=80
            else:
                threshold_percentile = self.args['random_walk_selection_percentile']
                
            # Analyze results efficiently
            influence_range_node = src.utils.analyze_walk_counts_efficient(
                walk_counts=walk_counts,
                threshold_percentile=threshold_percentile,
                min_visits=1
            )
            
            # Print some statistics
            total_influenced = sum(len(nodes) for nodes in influence_range_node.values())
            avg_influenced = total_influenced / len(influence_range_node)
            self.logger.info(f"Average number of influenced nodes: {avg_influenced:.2f}")
            
            # take the values() of influenced_range_node without repeat
            influence_range_node = set([item for sublist in influence_range_node.values() for item in sublist])
            self.logger.info(f"With {len(influence_range_node)} highly influenced nodes")
            
        if isinstance(influence_range_node, set):
            influence_range_node = list(influence_range_node)
            
        before_influenced_loss =  node_classifier.gen_grad_by_batch(data=None, data_sampler='neighbor', optimizer=None,
                                                            device=None, input_nodes=influence_range_node, batch_size = len(influence_range_node))
        
        node_classifier.swap_x_edge_index(node_classifier.data.edge_index_unlearn, node_classifier.data.edge_index, node_classifier.data.x_unlearn, node_classifier.data.x)
        
        after_influenced_loss =  node_classifier.gen_grad_by_batch(data=None, data_sampler='neighbor', optimizer=None,
                                                            device=None, input_nodes=influence_range_node, batch_size = len(influence_range_node))
        
        node_classifier.swap_x_edge_index(new_edge_index=node_classifier.data.old_edge_index, old_edge_index=node_classifier.data.edge_index, 
                                       new_x = node_classifier.data.old_x, old_x = node_classifier.data.x)
        
        # solve the equation to get the update
        # self.logger.info("unlearned_loss, before_influenced_loss, after_influenced_loss:", unlearned_loss, before_influenced_loss, after_influenced_loss)
        if(unlearned_loss == None):
            loss = before_influenced_loss - after_influenced_loss
        else:
            loss = unlearned_loss + before_influenced_loss - after_influenced_loss
        
        # Get parameters that require gradients
        params = [p for p in node_classifier.model.parameters() if p.requires_grad]
        
        # Initialize accumulator for HVP
        hvp_accum = [torch.zeros_like(p) for p in params]
        
        # Calculate first-order gradients
        # hvp_updates = self.stochastic_hvp(node_classifier, None, sampling_K, damping, sampling_method = sampling_method)  grad_Lu = torch.autograd.grad(loss, params, create_graph=True)
        
        gradients = torch.autograd.grad(loss, params, create_graph=True)
        
        # perform hvp
        hvp_updates = self.hvp_op(params, gradients, hvp_accum, damping = 1.0, sampling_K = 1)
        
        # update the parameters
        params_change = [h_est * learning_rate for h_est in hvp_updates]
       
        self.logger.info("learning_rate: %.5f" % learning_rate)
        self.logger.info("params_change norm: %s" % (torch.norm(torch.cat([item.flatten() for item in params_change]), 2)))
         
        params_esti   = [p1 + p2 for p1, p2 in zip(params_change, params)]
        
        # return the updated parameters
        return params_esti, params_change
    
    def katz_index_update(self, node_classifier, learning_rate = 1):
        # find influence range based on katz index
        # katz_index_matrix = self.katz_index_matrix
        # print("katz_index_matrix.shape:", katz_index_matrix.shape)
        
        # calculate the gradient of the unlearned nodes
        unlearned_loss = self.target_model.gen_grad_by_batch(data=self.data, data_sampler='neighbor', optimizer=None,
                                                            device=None, input_nodes=self.deleted_nodes, batch_size = len(self.deleted_nodes))
        
        influenced_nodes, visit_counts = src.utils.random_walk_influence(
            edge_index = node_classifier.data.edge_index,
            num_nodes = node_classifier.data.num_nodes,
            deleted_nodes = self.deleted_nodes,
            num_walks=20,
            walk_length=10,
            p=0.2
        )
        
        influence_range_node, threshold = src.utils.filter_by_visit_threshold(visit_counts)

        print(f"Found {len(influenced_nodes)} influenced nodes")
        print(f"With {len(influence_range_node)} highly influenced nodes")

        # calculate the gradient of the influenced nodes
        # influence_range_node = src.utils.calculate_katz_for_node_pairs_gpu_fullbatch(edge_index = node_classifier.data.edge_index, num_nodes = node_classifier.data.num_nodes,
        #                                                                    deleted_nodes = self.deleted_nodes, beta=0.1, max_iter=10, tolerance=1e-6)
        
        #scores_matrix, node_to_row = src.utils.calculate_katz_scores_gpu_batched(edge_index = node_classifier.data.edge_index, num_nodes = node_classifier.data.num_nodes,
        #                                                                        deleted_nodes = self.deleted_nodes)
        #influence_range_node, select_thres = src.utils.filter_sparse_columns_by_percentile(scores_matrix, percentile = 20)
        
        # influence_range_node, select_thres = src.utils.select_influence_range_by_katz(katz_index = katz_index_matrix, deleted_nodes = self.deleted_nodes, thres = 20)
        # remained_nodes = np.setdiff1d(np.arange(self.data.num_nodes), self.deleted_nodes)
        # influence_range_node, score = src.utils.select_nodes_by_katz_threshold(data = node_classifier.data, source_nodes = self.deleted_nodes, 
        #                                                                       target_nodes = remained_nodes, threshold = 1e-8, beta=0.1, max_path_length=10, batch_size=100)
        # influence_range_node = src.utils.select_nodes_by_katz_threshold(edge_index = node_classifier.data.edge_index, num_nodes = node_classifier.data.num_nodes,
        #                                                                deleted_nodes = self.deleted_nodes, remained_nodes = remained_nodes, threshold = 1e-5, 
        #                                                                beta=0.9, max_iter=10, tolerance=1e-6, batch_size=100)
        
        # threshold_bfs = 10
        # influence_range_node = src.utils.find_nodes_within_path_length(edge_index = node_classifier.data.edge_index, num_nodes = node_classifier.data.num_nodes, 
        #                                                                source_nodes = self.deleted_nodes, threshold = threshold_bfs)
        
        # self.logger.info("len(influence_range_node): %d, select_thres: %4f" % (len(influence_range_node), select_thres))
        
        # if influence_range_node is not list, turn it into list
        if isinstance(influence_range_node, set):
            influence_range_node = list(influence_range_node)
            
        before_influenced_loss =  node_classifier.gen_grad_by_batch(data=None, data_sampler='neighbor', optimizer=None,
                                                            device=None, input_nodes=influence_range_node, batch_size = len(influence_range_node))
        
        node_classifier.swap_x_edge_index(node_classifier.data.edge_index_unlearn, node_classifier.data.edge_index, node_classifier.data.x_unlearn, node_classifier.data.x)
        
        after_influenced_loss =  node_classifier.gen_grad_by_batch(data=None, data_sampler='neighbor', optimizer=None,
                                                            device=None, input_nodes=influence_range_node, batch_size = len(influence_range_node))
        
        node_classifier.swap_x_edge_index(new_edge_index=node_classifier.data.old_edge_index, old_edge_index=node_classifier.data.edge_index, 
                                       new_x = node_classifier.data.old_x, old_x = node_classifier.data.x)
        
        # solve the equation to get the update
        print("unlearned_loss, before_influenced_loss, after_influenced_loss:", unlearned_loss, before_influenced_loss, after_influenced_loss)
        loss = unlearned_loss + before_influenced_loss - after_influenced_loss
        
        # Get parameters that require gradients
        params = [p for p in node_classifier.model.parameters() if p.requires_grad]
        
        # Initialize accumulator for HVP
        hvp_accum = [torch.zeros_like(p) for p in params]
        
        # Calculate first-order gradients
        
        gradients = torch.autograd.grad(loss, params, create_graph=True)
        
        # perform hvp
        hvp_updates = self.hvp_op(params, gradients, hvp_accum, sampling_K = 1)
        
        # update the parameters
        params_change = [h_est * learning_rate for h_est in hvp_updates]
       
        self.logger.info("learning_rate: %.5f" % learning_rate)
        self.logger.info("params_change norm: %s" % (torch.norm(torch.cat([item.flatten() for item in params_change]), 2)))
         
        params_esti   = [p1 + p2 for p1, p2 in zip(params_change, params)]
        
        # return the updated parameters
        return params_esti, params_change
    
    def approxi(self, remained = 0, hessian_approximation_method = "hvpgif", add_noise = True, attack_prep = True):
        '''
        res_tuple == (grad_all, grad1, grad2)
        or res_tuple == (grad_all, grad1, grad2, grad_remained) when remained == True
        '''
        
        #params_esti =  [p for p in self.target_model.model.parameters() if p.requires_grad]
        #init_F1 = self.target_model.evaluate_unlearn_F1(params_esti, data_sampler = self.args['data_sampler'])
        #self.logger.info("init F1: %.4f" % init_F1)
        
        start_time = time.time()
        
        torch.cuda.empty_cache()
        torch.cuda.reset_accumulated_memory_stats()
        start_mem = torch.cuda.memory_allocated()
        
        high_influence_nodes = None
        if(hessian_approximation_method == "adaptiveselection"):
            if self.args["unlearn_task"] == 'node':
                specific_nodes = self.deleted_nodes
            if self.args["unlearn_task"] == 'edge':
                specific_nodes = self.influence_nodes
            hessian_approximation_method, importance_sampling_value, random_walk_value, high_influence_nodes = src.utils.select_approxi_method_with_global_randomwalk_stats(dataset_name = self.args['dataset_name'], deleted_nodes = specific_nodes, 
                                                                                                                                                                            importance_sampling_K = self.args['hvp_sampling_K'], importance_sampling_N = self.args['hvp_sampling_size'])
            self.logger.info("hessian_approximation_method: %s, importance_sampling_value:%d, random_walk_value: %d" % (hessian_approximation_method, importance_sampling_value, random_walk_value))

        if(hessian_approximation_method == "hvpgif"):
            if(self.args["gif_stochastic"] == False):
                # directly sample dataset to perform hvp in IDEA
                params_esti, params_change   = self.hvp_gif( remained = remained, learning_rate=self.args["train_lr"], sampling_dataset=self.args['gif_sampling_dataset'] )
            elif(self.args["gif_stochastic"] == True):
                # perform stochastic sampling based on hvp in IDEA
                params_esti, params_change   = self.hvp_gif_stochastic(remained = remained, learning_rate=self.args["train_lr"])
        elif(hessian_approximation_method == "stochastichvp"):
            params_esti, params_change   = self.apply_stochastic_update( node_classifier = self.target_model, 
                                                                        learning_rate=1, sampling_K=self.args["hvp_sampling_K"], sampling_method = "random")
        elif(hessian_approximation_method == "importancesampling"):
            params_esti, params_change   = self.apply_stochastic_update( node_classifier = self.target_model, 
                                                                    learning_rate=1, sampling_K=self.args["hvp_sampling_K"], sampling_method = self.args["importance_measurement"])
        elif(hessian_approximation_method == "katzindex"):
            params_esti, params_change   = self.katz_index_update( node_classifier = self.target_model )
        elif(hessian_approximation_method == "randomwalk"):
            params_esti, params_change   = self.random_walk_update( node_classifier = self.target_model, high_influence_nodes = high_influence_nodes)
        elif(hessian_approximation_method == "lbfgs"):
            params_esti, params_change   = self.apply_LBFGS_update( node_classifier = self.target_model, data = self.data)
        else:
            params_esti = None
            
        #second_F1 = self.target_model.evaluate_unlearn_F1(params_esti, data_sampler = self.args['data_sampler'])
        #self.logger.info("second F1: %.4f" % second_F1)
            
        end_time = time.time()
        peak_mem = torch.cuda.max_memory_allocated()
        
        memory_used = (peak_mem - start_mem) / 1024**2
        self.logger.info(f"Peak memory used: {memory_used:.2f} MB")
        self.params_esti_bound = params_esti
        
        # add Gaussian Noise
        if(add_noise == True):
            
            total_size = sum([torch.sum(torch.ones(item.size())) for item in params_esti])
            if(self.args['target_model'] == 'SimpleGCN'):
                m = self.samples_to_be_unlearned
                # certification_alpha1 = (self.args['c'] + m * self.args['l'] + (m ** 2 * self.args['l'] ** 2 + 8 * self.args['lambda'] * (len(self.train_indices)-m) * self.args['c']) ** 0.5) / (self.args['lambda'])
                # certification_alpha1 = (self.args['c2'] * self.args['train_alpha'] + m * self.args['l'] + (m ** 2 * self.args['l'] ** 2 + 8 *  (len(self.train_indices)-m) * self.args['c'] * self.args['lambda'] * self.args['train_alpha']) ** 0.5) / (self.args['lambda'] * self.args['train_alpha'])
                certification_alpha1 = (m * self.args['l'] + (m ** 2 * self.args['l'] ** 2 + 8 *  (len(self.train_indices)-m) * self.args['c'] * self.args['lambda'] * self.args['train_alpha']) ** 0.5) / (self.args['lambda'] * self.args['train_alpha'])
                params_change_flatten = [item.flatten() for item in params_change]
            
                certification_alpha2 = torch.norm(torch.cat(params_change_flatten), 2) # /len(self.train_indices)
                #certification_alpha2 = 0.001
                epsilon, delta = 1, 1e-3    
                gaussian_std = ((certification_alpha1 + certification_alpha2)/epsilon) * math.sqrt(2 * math.log(1.25/delta)) / math.sqrt(total_size)
            else:
                gaussian_std = 0.001
            self.logger.info("Add Gaussian noise with std %.5f and mean %.4f. Total size of params: %d." % (gaussian_std , self.args['gaussian_mean'], total_size))
            gaussian_noise = [(torch.randn(item.size(), device=item.device) * gaussian_std + self.args['gaussian_mean']) for item in params_esti]
            params_esti = [item1 + item2 for item1, item2 in zip(gaussian_noise, params_esti)]

        # test_F1 = self.evaluate(run, data_sampler = self.args['data_sampler']) self.target_model.evaluate_unlearn_F1(params_esti)
        unlearn_F1 = self.target_model.evaluate_unlearn_F1(params_esti, data_sampler = self.args['data_sampler'], attack_prep = attack_prep)
        self.params_esti = params_esti
        self.logger.info("unlearn F1: %.4f" % unlearn_F1)
        
        return end_time - start_time, unlearn_F1, params_change, memory_used, hessian_approximation_method

    def alpha_computation_LGU(self, params_change):
        
        self.logger.info("==== alpha computation LGU ====")
        
        m = self.samples_to_be_unlearned
        t = self.influence_nodes.shape[0]
        
        #self.certification_alpha1 = (self.args['c2'] * self.args['train_alpha'] + m * self.args['l'] + (m ** 2 * self.args['l'] ** 2 + 8 *  (len(self.train_indices)-m) * self.args['c'] * self.args['lambda'] * self.args['train_alpha']) ** 0.5) / (self.args['lambda'] * self.args['train_alpha'])
        #self.logger.info("m:", m, "self.args['l']:", self.args['l'], "len(self.train_indices):", len(self.train_indices), "self.args['c']:", self.args['c'], "self.args['lambda']:", self.args['lambda'], "self.args['train_alpha']:", self.args['train_alpha'])
        self.certification_alpha1 = (m * self.args['l'] + (m ** 2 * self.args['l'] ** 2 + 8 *  (len(self.train_indices)-m) * self.args['c'] * self.args['lambda'] * self.args['train_alpha']) ** 0.5) / (self.args['lambda'] * self.args['train_alpha'])
        params_change_flatten = [item.flatten() for item in params_change]
        
        self.certification_alpha2 = torch.norm(torch.cat(params_change_flatten), 2)
        
        self.logger.info("Certification related stats:  ")
        self.logger.info("certification_alpha1 (bound): %s" % self.certification_alpha1)
        self.logger.info("certification_alpha2 (l2 of modification): %s" % self.certification_alpha2)
        total_bound = self.certification_alpha1 + self.certification_alpha2
        self.logger.info("total bound given by alpha1 + alpha2: %s" % total_bound)

        # bound given by certified edge
        certified_edge_bound = self.certification_alpha2 ** 2 * self.args['M'] / self.args['l']
        self.logger.info("data-dependent bound given by certified edge: %s" % certified_edge_bound)

        # worset bound given by certified edge  
        certified_edge_worst_bound = self.args['M'] * (self.args['gamma_2'] ** 2) * (self.args['c1'] ** 2) * (t ** 2) / ((self.args['lambda_edge_unlearn'] ** 4) * len(self.train_indices))
        self.logger.info("worst bound given by certified edge: %s" % certified_edge_worst_bound)
        
        # total_bound, certified_edge_bound, certified_edge_worst_bound = -1, -1, -1
        # test_F1_retrain, retraining_time = -1, -1
        # actual_param_difference = -1
        
        # continue optimizing the model with data already updated
        if(self.args['retrain_model_for_cmp'] == False):
            return total_bound, certified_edge_bound, certified_edge_worst_bound, actual_param_difference, test_F1_retrain, retraining_time
            # self.target_model.train_model_continue((self.deleted_nodes, self.feature_nodes, self.influence_nodes))
        else:
            assert self.args['unlearn_method'] not in ["retrain"]
            if(self.load_retrain_model() == False):
                retraining_time = self._train_model(run = -1, data_sampler = self.args['data_sampler'], modify_data_to_remained = True)
            else:
                retraining_time = -1  
            # self.target_model.retrain_model((self.deleted_nodes, self.feature_nodes, self.influence_nodes))
            
        # actual difference
        original_params = [p.flatten() for p in self.params_esti_bound]
        retraining_model_params = [p.flatten() for p in self.target_model.model.parameters() if p.requires_grad] # how can this be called retraining model??
        actual_param_difference = torch.norm((torch.cat(original_params) - torch.cat(retraining_model_params)), 2).detach()
        self.logger.info("actual params difference: %s" % actual_param_difference)

        # test_F1_retrain = self.target_model.evaluate_unlearn_F1([p for p in self.target_model.model.parameters() if p.requires_grad])
        
        if(self.args["evaluation"] == True):    
            test_F1_retrain = self.evaluate(run = -1, data_sampler = self.args['data_sampler'], attack_prep = False)
        else:
            test_F1_retrain = -1
        self.logger.info("retrain f1 score: %s" % test_F1_retrain)

        return total_bound, certified_edge_bound, certified_edge_worst_bound, actual_param_difference.cpu().numpy(), test_F1_retrain, retraining_time
        
    def alpha_computation_gif(self, params_change):

        # bound given by alpha 1 + alpha 2
        m = self.samples_to_be_unlearned
        t = self.influence_nodes.shape[0]
        
        # len(self.train_indices): m
        # m: |V|
        
        #self.certification_alpha1 = (m * self.args['l'] + (m ** 2 + self.args['l'] ** 2 + 4 * self.args['lambda'] * len(self.train_indices) * t * self.args['c']) ** 0.5) / (self.args['lambda'] * len(self.train_indices))
        #params_change_flatten = [item.flatten() for item in params_change]        
        #self.certification_alpha2 = torch.norm(torch.cat(params_change_flatten), 2)
        
        #self.certification_alpha1 = (m * self.args['l'] + (m ** 2 * self.args['l'] ** 2 + 4 * self.args['lambda'] * len(self.train_indices) * t * self.args['c']) ** 0.5) / (self.args['lambda'] * len(self.train_indices))
        self.certification_alpha1 = (self.args['c'] + m * self.args['l'] + (m ** 2 * self.args['l'] ** 2 + 8 * self.args['lambda'] * (len(self.train_indices)-m) * self.args['c']) ** 0.5) / (self.args['lambda'])
        params_change_flatten = [item.flatten() for item in params_change]
        
        self.certification_alpha2 = torch.norm(torch.cat(params_change_flatten), 2)/len(self.train_indices)

        self.logger.info("Certification related stats:  ")
        self.logger.info("certification_alpha1 (bound): %s" % self.certification_alpha1)
        self.logger.info("certification_alpha2 (l2 of modification): %s" % self.certification_alpha2)
        total_bound = self.certification_alpha1 + self.certification_alpha2
        self.logger.info("total bound given by alpha1 + alpha2: %s" % total_bound)

        # bound given by certified edge
        certified_edge_bound = self.certification_alpha2 ** 2 * self.args['M'] / self.args['l']
        self.logger.info("data-dependent bound given by certified edge: %s" % certified_edge_bound)

        # worset bound given by certified edge  
        certified_edge_worst_bound = self.args['M'] * (self.args['gamma_2'] ** 2) * (self.args['c1'] ** 2) * (t ** 2) / ((self.args['lambda_edge_unlearn'] ** 4) * len(self.train_indices))
        self.logger.info("worst bound given by certified edge: %s" % certified_edge_worst_bound)
        
        # recover the originally trained model
        idx = 0
        for p in self.target_model.model.parameters():
            p.data = self.originally_trained_model_params[idx].clone()
            idx = idx + 1
        
        actual_param_difference, test_F1_retrain, retraining_time = -1, -1, -1
        
        if(self.args['retrain_model_for_cmp'] == False):
            return total_bound, certified_edge_bound, certified_edge_worst_bound, actual_param_difference, test_F1_retrain, retraining_time
            # self.target_model.train_model_continue((self.deleted_nodes, self.feature_nodes, self.influence_nodes))
        else:
            assert self.args['unlearn_method'] not in ["retrain"]
            if(self.load_retrain_model() == False):
                retraining_time = self._train_model(run = -1, data_sampler = self.args['data_sampler'], modify_data_to_remained = True)
            else:
                retraining_time = -1
                            
        # actual difference
        original_params = [p.flatten() for p in self.params_esti]
        retraining_model_params = [p.flatten() for p in self.target_model.model.parameters() if p.requires_grad] # how can this be called retraining model??
        actual_param_difference = torch.norm((torch.cat(original_params) - torch.cat(retraining_model_params)), 2).detach()
        self.logger.info("actual params difference: %s" % actual_param_difference)

        # test_F1_retrain = self.target_model.evaluate_unlearn_F1([p for p in self.target_model.model.parameters() if p.requires_grad])
        
        if(self.args["evaluation"] == True):
            self.logger.info("===== alpha_computation_gif: RETRAIN MODEL =====")
            test_F1_retrain = self.evaluate(run = -1, data_sampler = self.args['data_sampler'], attack_prep = False)
        else:
            test_F1_retrain = -1
        self.logger.info("retrain f1 score: %s" % test_F1_retrain)

        return total_bound.cpu().numpy(), certified_edge_bound.cpu().numpy(), certified_edge_worst_bound, actual_param_difference.cpu().numpy(), test_F1_retrain, retraining_time

    # Method 1: Using torch.autograd.functional.hessian directly
    def compute_hessian_method1(model, loss_fn, inputs):
        """
        Calculate Hessian using torch.autograd.functional.hessian
        """
        def func(x):
            return loss_fn(model(x))
        
        hess = hessian(func, inputs)
        return hess

    # Method 2: Manual computation using jacobian of gradient
    def compute_hessian_method2(loss, model_params):
        """
        Calculate Hessian manually by computing gradients twice
        """
        # First backward pass - get gradients
        grads = grad(loss, model_params, create_graph=True)
        
        # Flatten gradients into a vector
        grad_vec = torch.cat([g.contiguous().view(-1) for g in grads])
        
        # Calculate Hessian
        hessian_rows = []
        for grad_idx in range(len(grad_vec)):
            # Second backward pass - get gradient of gradient
            grad2 = grad(grad_vec[grad_idx], model_params, retain_graph=True)
            hessian_row = torch.cat([g.contiguous().view(-1) for g in grad2])
            hessian_rows.append(hessian_row)
        
        return torch.stack(hessian_rows)

    # Method 3: Using vector-Jacobian product (vJP) for memory efficiency
    def compute_hessian_method3(loss, params, create_graph=False):
        """
        Calculate Hessian using vector-Jacobian products
        More memory efficient for large models
        """
        # Get gradients
        grads = grad(loss, params, create_graph=True)
        
        # Flatten parameters and gradients
        flat_params = torch.cat([p.view(-1) for p in params])
        flat_grads = torch.cat([g.view(-1) for g in grads])
        num_params = len(flat_params)
        
        # Initialize Hessian matrix
        H = torch.zeros(num_params, num_params)
        
        # Compute each row of the Hessian
        for idx in range(num_params):
            # Get grad of grad
            grad2 = grad(flat_grads[idx], params, retain_graph=True)
            flat_grad2 = torch.cat([g.contiguous().view(-1) for g in grad2])
            H[idx] = flat_grad2
            
        return H

    # compute actual Hessian matrix, same as Method 2
    def eval_hessian(self, loss_grad, model):
        cnt = 0
        for g in loss_grad:
            g_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([g_vector, g.contiguous().view(-1)])
            cnt = 1
        l = g_vector.size(0)
        hessian = torch.zeros(l, l).to(loss_grad[0].device)
        for idx in range(l):
            grad2rd = torch.autograd.grad(g_vector[idx], model.parameters(), create_graph=True)
            cnt = 0
            for g in grad2rd:
                g2 = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
                cnt = 1
            hessian[idx] = g2
        return hessian

    def hvps(self, grad_all, model_params, h_estimate):
        element_product = 0
        for grad_elem, v_elem in zip(grad_all, h_estimate):
            element_product += torch.sum(grad_elem * v_elem)
        
        # print("model_params: ", len(model_params))
        return_grads = grad(element_product,model_params,create_graph=True)
        return return_grads
    
    def writer_to_csv(self, writing_list, name="unlearning_results"):

        csv_file_path = self.args['file_name'] + ".csv"
        
        if not os.path.exists(csv_file_path):
            df = pd.DataFrame(columns=["dataset", "model", "train_ratio", "unlearn_task", "unlearn_ratio", "unlearn_method", "hessian_approxi_method", "scale", "iteration", \
                                       "f1_score_avg", "f1_score_std", "training_time_avg", \
                                        "training_time_std", "f1_score_unlearn_avg", "f1_score_unlearn_std", \
                                            "unlearning_time_avg", "unlearning_time_std", "my_bound_avg", \
                                                "my_bound_std", "certified_edge_bound_avg", "certified_edge_std", \
                                                    "certified_edge_worst_bound_avg", "certified_edge_worst_bound_std", \
                                                        "actual_diff_avg", "actual_diff_std", "f1_retrain_avg", "f1_retrain_std",\
                                                            #"grad_all_memory(MB)", "grad1_memory(MB)", "grad2_memory(MB)", 
                                                                "memory_used_avg", "memory_used_std", "retraining_time_avg", "retraining_time_std", ])
            df.to_csv(csv_file_path, index=False)

        df = pd.read_csv(csv_file_path)
        new_row = {"dataset": self.args['dataset_name'], "model": self.args['target_model'], "train_ratio":1-self.args['test_ratio'], \
                   "unlearn_task": self.args['unlearn_task'], "unlearn_ratio": self.args['unlearn_ratio'], \
                    "unlearn_method": self.args['unlearn_method'], "hessian_approxi_method": writing_list[11], "scale":self.args["scale"], "iteration":self.args['iteration'], \
                    "f1_score_avg": writing_list[0][0], "f1_score_std": writing_list[0][1], "training_time_avg": writing_list[1][0], \
                    "training_time_std": writing_list[1][1], "f1_score_unlearn_avg": writing_list[2][0], "f1_score_unlearn_std": writing_list[2][1], \
                    "unlearning_time_avg": writing_list[3][0], "unlearning_time_std": writing_list[3][1], "my_bound_avg": writing_list[4][0], \
                    "my_bound_std": writing_list[4][1], "certified_edge_bound_avg": writing_list[5][0], "certified_edge_std": writing_list[5][1], \
                    "certified_edge_worst_bound_avg": writing_list[6][0], "certified_edge_worst_bound_std": writing_list[6][1], \
                    "actual_diff_avg": writing_list[7][0], "actual_diff_std": writing_list[7][1], "f1_retrain_avg": writing_list[8][0], "f1_retrain_std": writing_list[8][1],\
                    # "grad_all_memory(MB)":self.target_model.memory_trace["grad_all"], "grad1_memory(MB)":self.target_model.memory_trace["grad1"], "grad2_memory(MB)":self.target_model.memory_trace["grad2"],
                    "memory_used_avg":writing_list[9][0], "memory_used_std":writing_list[9][1], "retraining_time_avg":writing_list[10][0], "retraining_time_std":writing_list[10][1]}
        
        # df = df.append(new_row, ignore_index=True)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(csv_file_path, index=False)
        
        self.logger.info("writing to %s done!" % csv_file_path)
