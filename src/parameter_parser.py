import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parameter_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=12345, help='control whether to use multiprocess')

    ######################### general parameters ################################
    parser.add_argument('--is_vary', type=bool, default=False, help='control whether to use multiprocess')
    parser.add_argument('--cuda', type=int, default=-1, help='specify gpu')
    parser.add_argument('--num_threads', type=int, default=1)
    
    ########################## unlearning task parameters ######################
    parser.add_argument('--dataset_name', type=str, default='citeseer',
                        choices=["cora", "citeseer", "pubmed", "CS", "Physics","ogbn-arxiv", "ogbn-products", "reddit", "flickr", "yelp", "ogbn-mag", "ogbn-papers100M"])
    parser.add_argument('--find_k_hops', type=str2bool, default=True, help='whether or not find the k hops and expand the influence range')
    parser.add_argument('--use_remained_info_for_update', type=int, default=0)
    parser.add_argument('--retrain_model_for_cmp', type=str2bool, default=False, help='whether or not retrain the model before unlearning for comparison.')
    parser.add_argument('--unlearn_method', type=str, default='retrain', choices=["retrain", "IDEA","none", "LGU"])
    parser.add_argument('--unlearn_task', type=str, default='edge', choices=["edge", "node", 'feature', 'partial_feature'])
    parser.add_argument('--unlearn_ratio', type=float, default=0.1)
    parser.add_argument('--unlearn_feature_partial_ratio', type=float, default=0.5)
    parser.add_argument('--unlearn_less_important_nodes', type=str2bool, default=False, help='whether or not unlearn less important nodes')
    parser.add_argument('--save_remained_data', type=str2bool, default=False, help='whether or not save remained dataset')
    parser.add_argument('--load_remained_data_for_retraining', type=str2bool, default=False, help='whether or not save remained dataset')

    ########################## training parameters ###########################
    parser.add_argument('--data_sampler', type=str, default="none", choices=["none", "neighbor", "cluster"], help='sampling dataloader type used in the exp.')
    parser.add_argument('--is_split', type=str2bool, default=True, help='splitting train/test data')
    parser.add_argument('--test_ratio', type=float, default=0.1) # originally it's 0.1 -> 0.37
    parser.add_argument('--measure_memory', type=str2bool, default=False)
    parser.add_argument('--use_test_neighbors', type=str2bool, default=True)
    parser.add_argument('--is_train_target_model', type=str2bool, default=True)
    parser.add_argument('--is_use_node_feature', type=str2bool, default=True)
    parser.add_argument('--target_model', type=str, default='GAT', choices=["GAT", 'MLP', "GCN", "GIN","SGC", "SAGE", "GCNNet","SimpleGCN"])
    parser.add_argument('--num_GCNlayer', type=int, default=2)
    parser.add_argument('--train_lr', type=float, default=0.01)
    parser.add_argument('--train_weight_decay', type=float, default=0)
    parser.add_argument('--num_epochs', type=int, default=500)  # 3000
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--evaluation', type=str2bool, default=True)
    parser.add_argument('--loss_func', type=str, default='scaled_sum', choices=['scaled_sum', 'sum'])
    parser.add_argument('--evaluate_model_on_unlearned_nodes', type=str2bool, default=False)

    ########################## GIF parameters ###########################
    parser.add_argument('--iteration', type=int, default=5)
    parser.add_argument('--scale', type=int, default=50)
    parser.add_argument('--damp', type=float, default=0.0)

    ########################## unlearning certification parameters ###########################
    parser.add_argument('--using_Dr', type=str2bool, default=True, help='Using L(Dr, theta_0) to update Hessian or not, i.e., L(Du, theta_0).')
    parser.add_argument('--hessian_approximation_method', type=str, default="hvpgif", choices=["hvpgif", "stochastichvp","lbfgs","importancesampling","katzindex","randomwalk","adaptiveselection"], help='hessian approximation methods used in the exp.')
    parser.add_argument('--gif_stochastic', type=str2bool, default=False, help='whether or not perform stochastic hvp when using the gif method.')
    parser.add_argument('--hvp_sampling_K', type=int, default=5, help='sampling K times for stochastic hvp and importance sampling.')
    parser.add_argument('--hvp_sampling_size', type=int, default=-1, help='the number of sampling nodes for stochastic hvp and importance sampling.')
    parser.add_argument('--gif_sampling_dataset', type=float, default=-1, help='the proportion of dataset used in IDEA.')
    parser.add_argument('--importance_measurement', type=str, default="degree", choices=["degree", "pagerank"], help='the importance measurement used in importance sampling.')
    parser.add_argument('--gaussian_mean', type=float, default=0.0)
    parser.add_argument('--gaussian_std', type=float, default=0.000)
    parser.add_argument('--random_walk_K', type=int, default=30, help = 'random walk K times for each node')
    parser.add_argument('--random_walk_l', type=int, default=100, help = 'random walk length.')
    parser.add_argument('--random_walk_selection_percentile', type=int, default=80)

    ########################## unlearning certification bound setting ###########################
    parser.add_argument('--l', type=float, default=0.1, help="lipschitz constant of the loss.")
    parser.add_argument('--lambda', type=float, default=0.05, help="(original) loss function is lambda-strongly convex.")  # 0.05 1.0 
    parser.add_argument('--c', type=float, default=3.0, help="numerical bound of the training loss regarding each sample.")  # 3.0 0.5
    parser.add_argument('--c2', type=float, default=5.0, help="numerical bound of the training gradient regarding each sample.")  
    parser.add_argument('--train_alpha', type=float, default=1e5, help="scale coefficient of the loss function.")  

    ########################## unlearning baselines certification bound setting ###########################
    parser.add_argument('--M', type=float, default=0.25, help="the loss is M - Lipschitz Hessian   in terms of w, i.e., gamma_1 in certified edge unlearning.")
    parser.add_argument('--c1', type=float, default=1.0, help="value of the derivative of loss is c1 bounded.")
    parser.add_argument('--lambda_edge_unlearn', type=float, default=0.05, help="regularization term weight - edge unlearning.")
    parser.add_argument('--gamma_2', type=float, default=1.0, help="lipschitz constant of first-order derivative of the loss - edge unlearning.")
    
    ########################## general setting ###########################
    parser.add_argument('--file_name', type=str, default="unlearning_results", help="file name for results.")
    parser.add_argument('--write', type=bool, default=True, help="write to keep results.")
    parser.add_argument('--save_original_model', type=str2bool, default=True, help="whether or not save original model.")
    parser.add_argument('--save_unlearned_model', type=str2bool, default=True, help="whether or not save unlearned model.")
    args = vars(parser.parse_args())

    return args
