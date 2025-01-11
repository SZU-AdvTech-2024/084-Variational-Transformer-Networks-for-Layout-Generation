import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
_rico25_labels = [
    "Text",
    "Image",
    "Icon",
    "Text Button",
    "List Item",
    "Input",
    "Background Image",
    "Card",
    "Web View",
    "Radio Button",
    "Drawer",
    "Checkbox",
    "Advertisement",
    "Modal",
    "Pager Indicator",
    "Slider",
    "On/Off Switch",
    "Button Bar",
    "Toolbar",
    "Number Stepper",
    "Multi-Tab",
    "Date Picker",
    "Map View",
    "Video",
    "Bottom Navigation",
]
class JSONImageDataset(Dataset):
    def __init__(self,json_folder):
        #self.json_files = [os.path.join(json_folder, f) for f in os.listdir(json_folder) if f.endswith('.json')]
        self.json_files=json_folder
    def __len__(self):
        return len(self.json_files)
    def __getitem__(self,idx):
        json_file=self.json_files[idx]
        base_path,_=os.path.splitext(json_file)
        image_file=base_path+'.png'
        with open(json_file,'r') as f:
            # print(json_file)
            data=json.load(f)
            image=Image.open(image_file)
            self.max_x,self.max_y=image.size
            # print(self.max_x)
            # print(self.max_y)
        labels=self.extract_nodes(data)
        # print(idx)
        
        if labels is not None:
            labels = torch.tensor(labels, dtype=torch.float32)
        return labels
    def extract_nodes(self, data):
        """递归提取JSON数据中的节点"""
        nodes = []
        # 如果数据是字典，处理其中的bounds和componentLabel
        if isinstance(data, dict):
            # 创建一个包含bounds、componentLabel和idx的列表
            node_info = []
            one_hot=[0]*25
            if 'bounds' in data:
                node_info.extend(data['bounds'])  # 添加bounds信息
                #node_info[0:4]=[0,0,0,0]
                node_info[0]/=self.max_x
                node_info[1]/=self.max_y
                node_info[2]/=self.max_x
                node_info[3]/=self.max_y
            if 'componentLabel' in data:
                one_hot[_rico25_labels.index(data['componentLabel'])]=1
                  # 添加componentLabel信息
            node_info.extend(one_hot)
                #node_info.extend([_rico25_labels.index(data['componentLabel'])+1])
            # else:
            #     node_info.extend([0])
                # node_info.append(idx)  # 添加当前节点的idx标识
            
            nodes.append(node_info)  # 将该节点信息加入nodes列表

            # 如果有子节点，递归处理子节点
            if 'children' in data:
                for child in data['children']:
                    nodes.extend(self.extract_nodes(child))
            if len(nodes) == 0:
                return None  # 如果没有有效的节点，返回None
        return nodes
    def is_last_element(self,data,item):
        index=data.index(item)
        if index==len(data)-1:
            return 1
        else:
            return 0
def collate_fn(batch):
    """
    自定义 collate_fn，用于将不同长度的张量补 0。
    1. 保证每个样本的 labels 都是二维张量，且 feature_dim 固定为 5。
    2. 对节点数量补齐到 batch 内的最大节点数。
    """


    # 按节点数量补齐（每个节点特征维度固定）
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=0.0)
    return padded_batch
# json_folder="/mnt/d/data/RICO/semantic_annotations"
# dataset=JSONImageDataset(json_folder=json_folder)
# from torch.utils.data import DataLoader

# dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
# for batch in dataloader:
#     print(batch)
#     break