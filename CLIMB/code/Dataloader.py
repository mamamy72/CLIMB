import os
from scipy.io import loadmat
from torch_geometric.data import Data
import numpy as np
from scipy.stats import zscore
import torch

def pairwise_pearson_tensor(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:

    """

        Function: Compute the Pearson correlation coefficients between all pairs of rows from two matrices to obtain a similarity matrix.

        Input: 1) A: A tensor matrix of shape [N, M].
               2) B: A tensor matrix of shape [N, M].

        Output: 1) corr: A similarity tensor matrix of shape [N, N].

    """

    A_mean = A.mean(dim=1, keepdim=True)
    B_mean = B.mean(dim=1, keepdim=True)

    A_centered = A - A_mean
    B_centered = B - B_mean

    A_std = A_centered.norm(dim=1, keepdim=True)
    B_std = B_centered.norm(dim=1, keepdim=True)

    cov = torch.matmul(A_centered, B_centered.T)

    std_product = A_std @ B_std.T + 1e-8

    corr = cov / std_product

    return corr

def Construct_multimodal_group_and_individual_graph_data(args):

    """

        Function: This function is used to load group and individual data while constructing multi-level graph structures.

        Input: 1) args: All configuration parameters

        Output: 1) group_graph_data: Group graph data containing group-level multimodal data
                2) individual_graph_data_dict: A dictionary of individual graph data containing all multi-level graph data of one subject across all scan time points

    """

    """ Load the medial wall file """
    medial_wall_path = args.medial_wall_path
    medial_wall_data = loadmat(medial_wall_path)
    medial_wall_data_arr = medial_wall_data['medial_lh']
    index = np.where(medial_wall_data_arr.flatten() == 1)[0]

    """ Load the vertex coordinates file """
    vertice_corrdinate_path = args.vertice_corrdinate_path
    vertice_data = loadmat(vertice_corrdinate_path)
    vertice_data_arr = vertice_data['pos'].T
    vertice_data_arr=vertice_data_arr[index]

    """ Load the 1-ring adj matrix file """
    cluster_neibor_path=args.adj_matrix_1ring_path
    cluster_neibor_arr=np.load(cluster_neibor_path)
    cluster_neibor_tensor=torch.from_numpy(cluster_neibor_arr)

    """ Load the group FCN feature and MSN feature matrix file """
    FCN_group_data_path = args.FCN_group_data_path
    FCN_group_data_arr = np.load(FCN_group_data_path)
    MSN_group_data_path = args.MSN_group_data_path
    MSN_group_data_arr = np.load(MSN_group_data_path)

    """ Load the group parcellation map file """
    group_parcellation_path = args.group_parcellation_path
    group_parcellation = loadmat(group_parcellation_path)
    group_parcellation_arr = group_parcellation['label_lh']
    group_parcellation_arr = group_parcellation_arr[index] - 1

    """ Load the preparcellation file for dimensionality reduction """
    FCN_preparcellation_path = args.FCN_preparcellation_path
    FCN_preparcellation = loadmat(FCN_preparcellation_path)
    MSN_preparcellation_path = args.MSN_preparcellation_path
    MSN_preparcellation = loadmat(MSN_preparcellation_path)
    FCN_anchor_label = FCN_preparcellation['label_lh']
    FCN_anchor_label = FCN_anchor_label[index].flatten()
    MSN_anchor_label = MSN_preparcellation['label_lh']
    MSN_anchor_label = MSN_anchor_label[index].flatten()

    """ Load individual scan data across multiple age time points and create multi-level graphs """

    individual_graph_data_dict={}

    for age in os.listdir(args.individual_data_save_dir):

        FCN_individual_data_arr=None
        FCN_num = 0

        for dir_name in os.listdir(f'{args.individual_data_save_dir}/{age}'):

            if dir_name == 'MSN':

                """ Load individual MSN data """

                area_path = f'{args.individual_data_save_dir}/{age}/{dir_name}/area_lh.mat'
                curv_path = f'{args.individual_data_save_dir}/{age}/{dir_name}/curv_lh.mat'
                myelin_path = f'{args.individual_data_save_dir}/{age}/{dir_name}/myelin_lh.mat'
                thickness_path = f'{args.individual_data_save_dir}/{age}/{dir_name}/thickness_lh.mat'
                area = loadmat(area_path)
                area_data_arr = area['lh_area']
                curv = loadmat(curv_path)
                curv_data_arr = curv['lh_attri']
                myelin = loadmat(myelin_path)
                myelin_data_arr = myelin['lh_attri']
                thickness = loadmat(thickness_path)
                thickness_data_arr = thickness['lh_attri']
                MSN_feature_arr = np.concatenate((area_data_arr, curv_data_arr, myelin_data_arr, thickness_data_arr),axis=1)
                MSN_feature_arr = MSN_feature_arr[index]
                MSN_feature_arr = zscore(MSN_feature_arr, axis=0, ddof=1)
                MSN_arr_500 = np.concatenate([np.mean(MSN_feature_arr[MSN_anchor_label == i], axis=0, keepdims=True) for i in range(1, 501)], axis=0)

            else:

                """ Load individual FCN data """

                dtseries_path = f'{args.individual_data_save_dir}/{age}/{dir_name}/dtseries_lh.mat'
                FC = loadmat(dtseries_path)

                if FCN_individual_data_arr is not None:
                    FC_feature_arr = FC['cifti_timecourse']
                    FC_feature_arr = FC_feature_arr[index]
                    FC_arr_500 = np.concatenate([np.mean(FC_feature_arr[FCN_anchor_label == i], axis=0, keepdims=True) for i in range(1, 501)], axis=0)
                    FCN_individual_data_arr += pairwise_pearson_tensor(torch.from_numpy(FC_feature_arr).to('cuda:0'),torch.from_numpy(FC_arr_500).to('cuda:0')).cpu().numpy()
                else:
                    FC_feature_arr = FC['cifti_timecourse']
                    FC_feature_arr = FC_feature_arr[index]
                    FC_arr_500 = np.concatenate([np.mean(FC_feature_arr[FCN_anchor_label == i], axis=0, keepdims=True) for i in range(1, 501)], axis=0)
                    FCN_individual_data_arr = pairwise_pearson_tensor(torch.from_numpy(FC_feature_arr).to('cuda:0'),torch.from_numpy(FC_arr_500).to('cuda:0')).cpu().numpy()

                FCN_num+=1

        FCN_individual_data_arr= FCN_individual_data_arr / FCN_num
        MSN_individual_data_arr = pairwise_pearson_tensor(torch.from_numpy(MSN_feature_arr).to('cuda:0'),torch.from_numpy(MSN_arr_500).to('cuda:0')).cpu().numpy()

        FCN_group_data_tensor = torch.from_numpy(FCN_group_data_arr)
        FCN_individual_data_tensor = torch.from_numpy(FCN_individual_data_arr)
        MSN_group_data_tensor = torch.from_numpy(MSN_group_data_arr)
        MSN_individual_data_tensor = torch.from_numpy(MSN_individual_data_arr)
        input_data_pos_tensor = torch.from_numpy(vertice_data_arr)
        edges = torch.nonzero((cluster_neibor_tensor == 1), as_tuple=False).t()
        FCN_individual_edge_sim = pairwise_pearson_tensor(FCN_individual_data_tensor.to('cuda:0'), FCN_individual_data_tensor.to('cuda:0')).cpu()
        MSN_individual_edge_sim = pairwise_pearson_tensor(MSN_individual_data_tensor.to('cuda:0'), MSN_individual_data_tensor.to('cuda:0')).cpu()

        FCN_individual_edge_feature = FCN_individual_edge_sim[cluster_neibor_tensor == 1].view(-1, 1)
        MSN_individual_edge_feature = MSN_individual_edge_sim[cluster_neibor_tensor == 1].view(-1, 1)

        group_label_tensor = torch.from_numpy(group_parcellation_arr).flatten().long()

        new_edge_index1=torch.concat((torch.arange(0,FCN_individual_data_tensor.shape[0]).view(1,-1),torch.arange(FCN_individual_data_tensor.shape[0],FCN_individual_data_tensor.shape[0]*2).view(1,-1)),dim=0)
        new_edge_index2 = torch.concat((torch.arange(FCN_individual_data_tensor.shape[0], FCN_individual_data_tensor.shape[0] * 2).view(1, -1),torch.arange(0, FCN_individual_data_tensor.shape[0]).view(1, -1)), dim=0)
        group_graph_data = Data(x=torch.cat((FCN_group_data_tensor,MSN_group_data_tensor),dim=0).float(), y=group_label_tensor,edge_index=edges)
        individual_graph_data_dict[str(age)] = Data(x=torch.cat((FCN_individual_data_tensor, MSN_individual_data_tensor), dim=0).float(), pos=input_data_pos_tensor.float(), edge_index=torch.cat((edges, edges + FCN_individual_data_tensor.shape[0], new_edge_index1,new_edge_index2), dim=1), edge_attr=torch.cat((FCN_individual_edge_feature, MSN_individual_edge_feature,torch.ones(size=(FCN_individual_data_tensor.shape[0], 1)),torch.ones(size=(FCN_individual_data_tensor.shape[0],1)))).float())

    return group_graph_data, individual_graph_data_dict

