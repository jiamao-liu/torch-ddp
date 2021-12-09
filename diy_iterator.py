import torch
from torch.utils.data import Dataset
import numpy
class Bucket_Iterator(Dataset):
    def __init__(self,data, sort_key='text_indices'):
        super(Bucket_Iterator, self).__init__()
        self.sort_key = sort_key
        self.data=[item for item in data if len(item[self.sort_key])<100]
        span_out_list,cattext_indices_list,polarity_list,undir_graph_data_list=self.pad_data(self.data)
        self.span_out_list=torch.tensor(span_out_list)
        self.cattext_indices_list=torch.tensor(cattext_indices_list)
        self.polarity_list=torch.tensor(polarity_list)
        self.undir_graph_data_list=torch.tensor(undir_graph_data_list)
    def pad_data(self,batch_data):
        span_out_list=[]
        cattext_indices_list=[]
        polarity_list=[]
        undir_graph_data_list=[]
        max_len_cat = max([len(t["cattext_indices"]) for t in batch_data])
        for item in batch_data:
            text_indices, cattext_indices, left_indices, aspect_indices, span_out, polarity, undir_graph_data = item['text_indices'], item['cattext_indices'], item['left_indices'], item['aspect_indices'], item['span_out'], item['polarity'], item['undir_dependency_graph']
            cattext_padding = [0] * (max_len_cat - len(cattext_indices))
            span_out_list.append(span_out)
            cattext_indices_list.append(cattext_indices + cattext_padding)
            polarity_list.append(polarity)
            undir_graph_data_list.append(numpy.pad(undir_graph_data, ((0, max_len_cat - len(undir_graph_data)), (0, max_len_cat - len(undir_graph_data))), 'constant'))
        return span_out_list,cattext_indices_list,polarity_list,undir_graph_data_list
    def __getitem__(self,idx):
        return self.span_out_list[idx],self.cattext_indices_list[idx],self.polarity_list[idx],self.undir_graph_data_list[idx]
    def __len__(self):
        return self.polarity_list.shape[0]