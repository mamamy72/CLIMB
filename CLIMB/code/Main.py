import argparse
from Dataloader import *
from Model import *
from Trainer import *

parser = argparse.ArgumentParser(description='CLIMB')

parser.add_argument(
    '--medial_wall_path', action="store", required=False,dest="medial_wall_path", default=r"F:\data\medial_wall_30k.mat",
    help='The file path of medial wall')
parser.add_argument(
    '--vertice_corrdinate_path', action="store", required=False,dest="vertice_corrdinate_path", default=r'F:\data\lh_vertice_matrix.mat',
    help='The file path of vertice corrdinates')
parser.add_argument(
    '--adj_matrix_1ring_path', action="store", required=False,dest="adj_matrix_1ring_path", default=r'F:\data\adj_matrix_1ring.npy',
    help='The file path of 1-ring neighborhood matrix')
parser.add_argument(
    '--FCN_preparcellation_path', action="store", required=False,
    default=r'F:\data\FCN_preparcellation_500.mat',dest="FCN_preparcellation_path",
    help='The folder path of FCN_anchor')
parser.add_argument(
    '--MSN_preparcellation_path', action="store", required=False,
    default=r'F:\data\MSN_preparcellation_500.mat',dest="MSN_preparcellation_path",
    help='The folder path of MSN_anchor')
parser.add_argument(
    '--group_parcellation_path', action="store", required=False,
    default=r'F:\data\MMP_180.mat',dest="group_parcellation_path",
    help='The file path of group reference parcellation map')
parser.add_argument(
    '--FCN_group_data_path', action="store", required=False,
    default='F:\data\FCN_group_mean_arr.npy',dest="FCN_group_data_path",
    help='The file path of group FCN matrix')
parser.add_argument(
    '--MSN_group_data_path', action="store", required=False,
    default='F:\data\MCN_group_mean_arr.npy',dest="MSN_group_data_path",
    help='The file path of group MSN matrix')
parser.add_argument(
    '--individual_data_save_dir', action="store", required=False,
    default='F:\Train_data_BCP\MNBCP000178',dest="individual_data_save_dir",
    help='The dir path of individual data')


args = parser.parse_args()

if __name__ == '__main__':

    group_multimodal_graph,individual_multimodal_graph_dict=Construct_multimodal_group_and_individual_graph_data(args)

    model=CLIMB(ages_list=[int(age) for age in os.listdir(args.individual_data_save_dir)],start_channels=500, in_channels=128, hidden_channels=64, out_channels=512, mid_channels=512, num_class=180)

    Train_Longitudinal(model, group_multimodal_graph, individual_multimodal_graph_dict, 1000, 10,f'F:/project/model_save/model_para_save.pth',device='cuda:0')

    model.load_state_dict(torch.load(f'F:/project/model_save/model_para_save.pth'))

    Test_Longitudinal(args, model, individual_multimodal_graph_dict,r'F:\project\parcellation')

    print('Finished')

