import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn

class my_model(nn.Module):

    def __init__(self, start_channels, in_channels, hidden_channels, out_channels, mid_channels, num_class):
        super(my_model, self).__init__()

        self.hidden_channels = hidden_channels
        self.output_channels = out_channels
        self.num_class=num_class

        ''' Module 1: Multi-layer Transformer Graph Conv Encoder '''

        self.graph_transformer_conv1 = torch_geometric.nn.TransformerConv(start_channels, in_channels // 2, heads=2, bias=True,
                                                        edge_dim=1)
        self.graph_transformer_conv2 = torch_geometric.nn.TransformerConv(in_channels, hidden_channels // 2, heads=2, bias=True,
                                                        edge_dim=1)
        self.graph_transformer_conv3 = torch_geometric.nn.TransformerConv(hidden_channels, out_channels // 2, heads=2, bias=True,
                                                        edge_dim=1)

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)

        ''' Module 2: Spatial Position Encoder '''

        self.Spatial_position_encoder=nn.Linear(3, out_channels)

        ''' Module 3: Vertex-level Modality Weight Encoder '''

        self.Vertex_level_modality_weight_encoder = nn.Sequential(
            nn.Linear(start_channels * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )

        ''' Module 4: Parcellation Classifier '''

        self.Parcellation_classifier = nn.Sequential(
            nn.Linear(out_channels, mid_channels, bias=True),
            nn.BatchNorm1d(mid_channels),
            nn.GELU(),
            nn.Linear(mid_channels, num_class, bias=True),
        )

    def forward(self,graph_data):

        """ Multi-level Transformer graph convolution is used to learn the correlation relationships among multimodal data. """

        x, edge_index, edge_attr, pos= graph_data.x, graph_data.edge_index, graph_data.edge_attr, graph_data.pos

        x1 = self.graph_transformer_conv1(x, edge_index, edge_attr)
        x1 = self.bn1(x1)
        x1 = F.gelu(x1)

        x1 = self.graph_transformer_conv2(x1, edge_index, edge_attr)
        x1 = self.bn2(x1)
        x1 = F.gelu(x1)

        x1 = self.graph_transformer_conv3(x1, edge_index, edge_attr)
        x1 = self.bn3(x1)
        x1 = F.gelu(x1)

        """ The spatial coordinate encoder is used to incorporate spatial encoding into features and provide positional information. """

        pos_emd=self.Spatial_position_encoder(pos)

        """ Vertex-level modality weighting module is used to learn region-heterogeneous modality weights for fully exploiting modality information in different regions. """

        weight=self.Vertex_level_modality_weight_encoder(torch.cat((x[:x.shape[0] // 2],x[x.shape[0] // 2:]),dim=1))

        """ Multi-modal data fusion. """

        fused_features = weight[:,0].unsqueeze(-1) * x1[:x1.shape[0] // 2] + weight[:,1].unsqueeze(-1) * x1[x1.shape[0] // 2:] + pos_emd

        """ Multi-modal data fusion. """

        assignment_output = self.Parcellation_classifier(fused_features)

        return assignment_output


class CLIMB(nn.Module):

    def __init__(self, ages_list, start_channels, in_channels, hidden_channels, out_channels, mid_channels, num_class):
        super(CLIMB, self).__init__()

        self.ages = ages_list

        ''' One Subject One Model: scan data from different time points of the same subject are encoded using a shared-parameter encoder. '''

        self.encoders = my_model(
                start_channels, in_channels, hidden_channels,
                out_channels, mid_channels, num_class
            )

    def forward(self, target_data_dict):

        """ Input: a dictionary containing multiple multi-level graph data; Output: a dictionary containing multiple assignment results. """

        assignment_outputs_dict = {}

        for age in self.ages:

            assignment_outputs_dict[str(age)] = self.encoders(target_data_dict[str(age)])

        return assignment_outputs_dict