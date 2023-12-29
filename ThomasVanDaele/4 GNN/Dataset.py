from typing import Callable, List, Optional

import pandas as pd
import torch
from torch_geometric.data import HeteroData, InMemoryDataset


class IdentityEncoder:
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)


# A mapping is applied to the customers and articles for training
# This function reverses the mapping so we can map the predictions back to the original ids
def reverse_mapping(path, index_col, **kwargs):
    df = pd.read_parquet(path, **kwargs)
    df = df.set_index(index_col)
    mapping = {i: index for i, index in enumerate(df.index.unique())}
    return mapping


# Load the node data from a parquet file
def load_node_parquet(path, index_col, encoders=None, **kwargs):
    df = pd.read_parquet(path, **kwargs)
    df = df.set_index(index_col)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapping


# Load the edge data from a parquet file
def load_edge_parquet(path, src_index_col, src_mapping, dst_index_col, dst_mapping,
                      encoders=None, **kwargs):
    df = pd.read_parquet(path, **kwargs)

    df[src_index_col] = df[src_index_col].map(src_mapping)
    df[dst_index_col] = df[dst_index_col].map(dst_mapping)

    df = df.sort_values(ascending=True, by=[src_index_col])
    # df = df.groupby([src_index_col, dst_index_col]).size().reset_index(name='count')

    # src = [src_mapping[index] for index in df[src_index_col]]
    # dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([df[src_index_col], df[dst_index_col]])

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr


class HMDataset(InMemoryDataset):
    r"""
    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self) -> List[str]:
        return ['customers.parquet', 'articles.parquet', 'train.parquet', 'test.parquet']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        # Didn't find a direct way to download them since I think you have to be logged in to Kaggle
        # To access the files
        print("Get the files from Kaggle")

    def process(self):
        data = HeteroData()

        customers_path = self.raw_paths[0]
        articles_path = self.raw_paths[1]
        train_path = self.raw_paths[2]
        test_path = self.raw_paths[3]

        # We can just use the entity encoder here since the Radek preprocessing already did most of the work
        article_x, article_mapping = load_node_parquet(articles_path, index_col='article_id', encoders={
            'product_code': IdentityEncoder(dtype=torch.int),
            'prod_name': IdentityEncoder(dtype=torch.int),
            'product_type_no': IdentityEncoder(dtype=torch.int),
            'product_group_name': IdentityEncoder(dtype=torch.int),
            'detail_desc': IdentityEncoder(dtype=torch.int)
        })
        customer_x, customer_mapping = load_node_parquet(customers_path, index_col='customer_id', encoders={
            'FN': IdentityEncoder(dtype=torch.int),
            'Active': IdentityEncoder(dtype=torch.int),
            'club_member_status': IdentityEncoder(dtype=torch.int),
            'fashion_news_frequency': IdentityEncoder(dtype=torch.int),
            'age': IdentityEncoder(dtype=torch.int),
            'postal_code': IdentityEncoder(dtype=torch.int)
        })

        # Have to create the customer nodes before the article nodes
        # Otherwise the edge_index will be flipped and the training code will not work

        # Next line is when we want to use customer embeddings
        # data['customer'].x = customer_x
        # Customer nodes without embeddings
        data['customer'].num_nodes = len(customer_x)
        # Next line is when we want to use article embeddings
        # data['article'].x = article_x
        # Article nodes without embeddings
        data['article'].num_nodes = len(article_x)

        edge_index, edge_label = load_edge_parquet(
            train_path,
            src_index_col='customer_id',
            src_mapping=customer_mapping,
            dst_index_col='article_id',
            dst_mapping=article_mapping,
            # encoders={'count': IdentityEncoder(dtype=torch.int)},
        )
        edge_test_index, edge_test_label = load_edge_parquet(
            test_path,
            src_index_col='customer_id',
            src_mapping=customer_mapping,
            dst_index_col='article_id',
            dst_mapping=article_mapping,
            # encoders={'count': IdentityEncoder(dtype=torch.int)},
        )

        data['customer', 'bought', 'article']['edge_index'] = edge_index
        # Edge embeddings
        # data['customer', 'bought', 'article'].edge_label = edge_label
        data['article', 'bought_by', 'customer']['edge_index'] = edge_index.flip([0])
        # Edge embeddings
        # data['article', 'bought_by', 'customer'].edge_label = edge_label.flip([0])

        # Test data
        data['customer', 'bought', 'article']['edge_label_index'] = edge_test_index

        # Print debug info
        print(data['customer'].num_nodes)
        print(data['article'].num_nodes)
        print(data['customer', 'bought', 'article'].edge_index.shape)
        print(data['customer', 'bought', 'article'].edge_label_index.shape)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        # Save the processed data so we don't have to process it again
        self.save([data], self.processed_paths[0])
