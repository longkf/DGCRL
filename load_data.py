import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
import os.path as osp
import torch.nn.functional as F


class DataSet(InMemoryDataset):
    def __init__(self, root):
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')

    def process(self):
        data = getPygData(osp.join(self.root, 'raw/DREAM5_NetworkInference_GoldStandard_Network1 - in silico.tsv'),
                          osp.join(self.root, 'raw/net1_expression_data_in_silico.tsv'))
        # data = getPygData(osp.join(self.root, 'raw/DREAM5_NetworkInference_GoldStandard_Network2 - S. aureus.tsv'),
        #                   osp.join(self.root, 'raw/net2_expression_data_s_aureus.tsv'))
        # data = getPygData(osp.join(self.root, 'raw/DREAM5_NetworkInference_GoldStandard_Network3 - E. coli.tsv'),
        #                   osp.join(self.root, 'raw/net3_expression_data_e_coli.tsv'))
        # data = getPygData(osp.join(self.root, 'raw/DREAM5_NetworkInference_GoldStandard_Network4 - S. cerevisiae.tsv'),
        #                   osp.join(self.root, 'raw/net4_expression_data_s_cerevisiae.tsv'))
        data_list = [data]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def getPygData(edge_path, feat_path):
    edge = pd.read_csv(edge_path, sep='\t', header=None,
                       names=['gene1', 'gene2', 'link'])
    edge_list = []
    GtoIdx = {'G{}'.format(x + 1): x for x in range(10000)}
    for idx, row in edge.iterrows():
        if row['link'] == 1:
            edge_list.append([GtoIdx[row['gene1']], GtoIdx[row['gene2']]])
    edge = torch.tensor(edge_list)

    feat = pd.read_csv(feat_path, sep='\t', header=0)
    genes = []
    for i in range(1, feat.shape[1] + 1):
        genes.append(feat['G{}'.format(i)].tolist())
    feat = torch.tensor(genes)
    feat = F.normalize(feat, dim=1)
    data = Data(x=feat, edge_index=edge.t(), edge_attr=None)
    return data


