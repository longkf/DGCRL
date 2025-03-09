import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
import os.path as osp
import torch.nn.functional as F


class DataSet_SC(InMemoryDataset):
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
        # data = getPygData(osp.join(self.root, './Lofgof Dataset/mESC/TFs+500/BL--network.csv'),
        #                   osp.join(self.root, './Lofgof Dataset/mESC/TFs+500/BL--ExpressionData.csv'))
        # data = getPygData(osp.join(self.root, './Lofgof Dataset/mESC/TFs+1000/BL--network.csv'),
        #                   osp.join(self.root, './Lofgof Dataset/mESC/TFs+1000/BL--ExpressionData.csv'))

        # data = getPygData(osp.join(self.root, './Non-Specific Dataset/hESC/TFs+500/BL--network.csv'),
        #                   osp.join(self.root, './Non-Specific Dataset/hESC/TFs+500/BL--ExpressionData.csv'))
        # data = getPygData(osp.join(self.root, './Non-Specific Dataset/hESC/TFs+1000/BL--network.csv'),
        #                   osp.join(self.root, './Non-Specific Dataset/hESC/TFs+1000/BL--ExpressionData.csv'))
        # data = getPygData(osp.join(self.root, './Non-Specific Dataset/hHEP/TFs+500/BL--network.csv'),
        #                   osp.join(self.root, './Non-Specific Dataset/hHEP/TFs+500/BL--ExpressionData.csv'))
        # data = getPygData(osp.join(self.root, './Non-Specific Dataset/hHEP/TFs+1000/BL--network.csv'),
        #                   osp.join(self.root, './Non-Specific Dataset/hHEP/TFs+1000/BL--ExpressionData.csv'))
        # data = getPygData(osp.join(self.root, './Non-Specific Dataset/mDC/TFs+500/BL--network.csv'),
        #                   osp.join(self.root, './Non-Specific Dataset/mDC/TFs+500/BL--ExpressionData.csv'))
        # data = getPygData(osp.join(self.root, './Non-Specific Dataset/mDC/TFs+1000/BL--network.csv'),
        #                   osp.join(self.root, './Non-Specific Dataset/mDC/TFs+1000/BL--ExpressionData.csv'))
        # data = getPygData(osp.join(self.root, './Non-Specific Dataset/mESC/TFs+500/BL--network.csv'),
        #                   osp.join(self.root, './Non-Specific Dataset/mESC/TFs+500/BL--ExpressionData.csv'))
        # data = getPygData(osp.join(self.root, './Non-Specific Dataset/mESC/TFs+1000/BL--network.csv'),
        #                   osp.join(self.root, './Non-Specific Dataset/mESC/TFs+1000/BL--ExpressionData.csv'))
        # data = getPygData(osp.join(self.root, './Non-Specific Dataset/mHSC-E/TFs+500/BL--network.csv'),
        #                   osp.join(self.root, './Non-Specific Dataset/mHSC-E/TFs+500/BL--ExpressionData.csv'))
        # data = getPygData(osp.join(self.root, './Non-Specific Dataset/mHSC-E/TFs+1000/BL--network.csv'),
        #                   osp.join(self.root, './Non-Specific Dataset/mHSC-E/TFs+1000/BL--ExpressionData.csv'))
        # data = getPygData(osp.join(self.root, './Non-Specific Dataset/mHSC-GM/TFs+500/BL--network.csv'),
        #                   osp.join(self.root, './Non-Specific Dataset/mHSC-GM/TFs+500/BL--ExpressionData.csv'))
        # data = getPygData(osp.join(self.root, './Non-Specific Dataset/mHSC-GM/TFs+1000/BL--network.csv'),
        #                   osp.join(self.root, './Non-Specific Dataset/mHSC-GM/TFs+1000/BL--ExpressionData.csv'))
        # data = getPygData(osp.join(self.root, './Non-Specific Dataset/mHSC-L/TFs+500/BL--network.csv'),
        #                   osp.join(self.root, './Non-Specific Dataset/mHSC-L/TFs+500/BL--ExpressionData.csv'))
        # data = getPygData(osp.join(self.root, './Non-Specific Dataset/mHSC-L/TFs+1000/BL--network.csv'),
        #                   osp.join(self.root, './Non-Specific Dataset/mHSC-L/TFs+1000/BL--ExpressionData.csv'))
        #
        # data = getPygData(osp.join(self.root, './Specific Dataset/hESC/TFs+500/BL--network.csv'),
        #                   osp.join(self.root, './Specific Dataset/hESC/TFs+500/BL--ExpressionData.csv'))
        # data = getPygData(osp.join(self.root, './Specific Dataset/hESC/TFs+1000/BL--network.csv'),
        #                   osp.join(self.root, './Specific Dataset/hESC/TFs+1000/BL--ExpressionData.csv'))
        data = getPygData(osp.join(self.root, './Specific Dataset/hHEP/TFs+500/BL--network.csv'),
                          osp.join(self.root, './Specific Dataset/hHEP/TFs+500/BL--ExpressionData.csv'))
        # data = getPygData(osp.join(self.root, './Specific Dataset/hHEP/TFs+1000/BL--network.csv'),
        #                   osp.join(self.root, './Specific Dataset/hHEP/TFs+1000/BL--ExpressionData.csv'))
        # data = getPygData(osp.join(self.root, './Specific Dataset/mDC/TFs+500/BL--network.csv'),
        #                   osp.join(self.root, './Specific Dataset/mDC/TFs+500/BL--ExpressionData.csv'))
        # data = getPygData(osp.join(self.root, './Specific Dataset/mDC/TFs+1000/BL--network.csv'),
        #                   osp.join(self.root, './Specific Dataset/mDC/TFs+1000/BL--ExpressionData.csv'))
        # data = getPygData(osp.join(self.root, './Specific Dataset/mESC/TFs+500/BL--network.csv'),
        #                   osp.join(self.root, './Specific Dataset/mESC/TFs+500/BL--ExpressionData.csv'))
        # data = getPygData(osp.join(self.root, './Specific Dataset/mESC/TFs+1000/BL--network.csv'),
        #                   osp.join(self.root, './Specific Dataset/mESC/TFs+1000/BL--ExpressionData.csv'))
        # data = getPygData(osp.join(self.root, './Specific Dataset/mHSC-E/TFs+500/BL--network.csv'),
        #                   osp.join(self.root, './Specific Dataset/mHSC-E/TFs+500/BL--ExpressionData.csv'))
        # data = getPygData(osp.join(self.root, './Specific Dataset/mHSC-E/TFs+1000/BL--network.csv'),
        #                   osp.join(self.root, './Specific Dataset/mHSC-E/TFs+1000/BL--ExpressionData.csv'))
        # data = getPygData(osp.join(self.root, './Specific Dataset/mHSC-GM/TFs+500/BL--network.csv'),
        #                   osp.join(self.root, './Specific Dataset/mHSC-GM/TFs+500/BL--ExpressionData.csv'))
        # data = getPygData(osp.join(self.root, './Specific Dataset/mHSC-GM/TFs+1000/BL--network.csv'),
        #                   osp.join(self.root, './Specific Dataset/mHSC-GM/TFs+1000/BL--ExpressionData.csv'))
        # data = getPygData(osp.join(self.root, './Specific Dataset/mHSC-L/TFs+500/BL--network.csv'),
        #                   osp.join(self.root, './Specific Dataset/mHSC-L/TFs+500/BL--ExpressionData.csv'))
        # data = getPygData(osp.join(self.root, './Specific Dataset/mHSC-L/TFs+1000/BL--network.csv'),
        #                   osp.join(self.root, './Specific Dataset/mHSC-L/TFs+1000/BL--ExpressionData.csv'))

        # data = getPygData(osp.join(self.root, './STRING Dataset/hESC/TFs+500/BL--network.csv'),
        #                   osp.join(self.root, './STRING Dataset/hESC/TFs+500/BL--ExpressionData.csv'))
        # data = getPygData(osp.join(self.root, './STRING Dataset/hESC/TFs+1000/BL--network.csv'),
        #                   osp.join(self.root, './STRING Dataset/hESC/TFs+1000/BL--ExpressionData.csv'))
        # data = getPygData(osp.join(self.root, './STRING Dataset/hHEP/TFs+500/BL--network.csv'),
        #                   osp.join(self.root, './STRING Dataset/hHEP/TFs+500/BL--ExpressionData.csv'))
        # data = getPygData(osp.join(self.root, './STRING Dataset/hHEP/TFs+1000/BL--network.csv'),
        #                   osp.join(self.root, './STRING Dataset/hHEP/TFs+1000/BL--ExpressionData.csv'))
        # data = getPygData(osp.join(self.root, './STRING Dataset/mDC/TFs+500/BL--network.csv'),
        #                   osp.join(self.root, './STRING Dataset/mDC/TFs+500/BL--ExpressionData.csv'))
        # data = getPygData(osp.join(self.root, './STRING Dataset/mDC/TFs+1000/BL--network.csv'),
        #                   osp.join(self.root, './STRING Dataset/mDC/TFs+1000/BL--ExpressionData.csv'))
        # data = getPygData(osp.join(self.root, './STRING Dataset/mESC/TFs+500/BL--network.csv'),
        #                   osp.join(self.root, './STRING Dataset/mESC/TFs+500/BL--ExpressionData.csv'))
        # data = getPygData(osp.join(self.root, './STRING Dataset/mESC/TFs+1000/BL--network.csv'),
        #                   osp.join(self.root, './STRING Dataset/mESC/TFs+1000/BL--ExpressionData.csv'))
        # data = getPygData(osp.join(self.root, './STRING Dataset/mHSC-E/TFs+500/BL--network.csv'),
        #                   osp.join(self.root, './STRING Dataset/mHSC-E/TFs+500/BL--ExpressionData.csv'))
        # data = getPygData(osp.join(self.root, './STRING Dataset/mHSC-E/TFs+1000/BL--network.csv'),
        #                   osp.join(self.root, './STRING Dataset/mHSC-E/TFs+1000/BL--ExpressionData.csv'))
        # data = getPygData(osp.join(self.root, './STRING Dataset/mHSC-GM/TFs+500/BL--network.csv'),
        #                   osp.join(self.root, './STRING Dataset/mHSC-GM/TFs+500/BL--ExpressionData.csv'))
        # data = getPygData(osp.join(self.root, './STRING Dataset/mHSC-GM/TFs+1000/BL--network.csv'),
        #                   osp.join(self.root, './STRING Dataset/mHSC-GM/TFs+1000/BL--ExpressionData.csv'))
        # data = getPygData(osp.join(self.root, './STRING Dataset/mHSC-L/TFs+500/BL--network.csv'),
        #                   osp.join(self.root, './STRING Dataset/mHSC-L/TFs+500/BL--ExpressionData.csv'))
        # data = getPygData(osp.join(self.root, './STRING Dataset/mHSC-L/TFs+1000/BL--network.csv'),
        #                   osp.join(self.root, './STRING Dataset/mHSC-L/TFs+1000/BL--ExpressionData.csv'))
        data_list = [data]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def getPygData(edge_path, feat_path):
    names = pd.read_csv(feat_path, header=0, usecols=[0])
    name2idx = {}
    for idx, name in names.iterrows():
        name2idx[name.item()] = idx
    edge_list = []
    edge = pd.read_csv(edge_path, header=0, names=['Gene1', 'Gene2'])
    for idx, row in edge.iterrows():
        edge_list.append([name2idx[row['Gene1']], name2idx[row['Gene2']]])
    edge = torch.tensor(edge_list).t()
    genes = []
    feat = pd.read_csv(feat_path, header=0)
    for i in range(0, feat.shape[0]):
        tmp = feat.iloc[i].tolist()[1:]
        genes.append(tmp)
    feat = torch.tensor(genes, dtype=torch.float32)
    feat = F.normalize(feat, dim=1)
    data = Data(x=feat, edge_index=edge, edge_attr=None)
    return data


if __name__ == '__main__':
    dataset = DataSet_SC('./data/single-cell')
    data = dataset[0]
    print(data)
