# Applied from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/lightgcn.py

from pathlib import Path

from matplotlib import pyplot as plt
from torch_geometric.nn import LightGCN
from torch_geometric.utils import degree
from tqdm import tqdm

from Dataset import *

# Select hardware device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
path = Path('../data/HMDataset')
dataset = (HMDataset(str(path)))
data = dataset[0]
num_customer, num_article = data['customer'].num_nodes, data['article'].num_nodes

# Convert heterogeneous graph to homogeneous graph
data = data.to_homogeneous().to(device)

batch_size = 8192
mask = data.edge_index[0] < data.edge_index[1]
train_edge_label_index = data.edge_index[:, mask]
train_loader = torch.utils.data.DataLoader(
    range(train_edge_label_index.size(1)),
    shuffle=True,
    batch_size=batch_size,
)

model = LightGCN(
    num_nodes=data.num_nodes,
    embedding_dim=64,
    num_layers=2,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train():
    total_loss = total_examples = 0

    for index in train_loader:
        # Sample positive and negative labels.
        pos_edge_label_index = train_edge_label_index[:, index]
        neg_edge_label_index = torch.stack([
            pos_edge_label_index[0],
            torch.randint(num_customer, num_customer + num_article,
                          (index.numel(),), device=device)
        ], dim=0)
        edge_label_index = torch.cat([
            pos_edge_label_index,
            neg_edge_label_index,
        ], dim=1)

        optimizer.zero_grad()
        pos_rank, neg_rank = model(data.edge_index, edge_label_index).chunk(2)

        loss = model.recommendation_loss(
            pos_rank,
            neg_rank,
            node_id=edge_label_index.unique(),
        )
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * pos_rank.numel()
        total_examples += pos_rank.numel()

    return total_loss / total_examples


@torch.no_grad()
def test(k: int):
    emb = model.get_embedding(data.edge_index)
    user_emb, book_emb = emb[:num_customer], emb[num_customer:]

    precision = recall = total_examples = 0
    for start in range(0, num_customer, batch_size):
        end = start + batch_size
        logits = user_emb[start:end] @ book_emb.t()

        # Exclude training edges:
        mask = ((train_edge_label_index[0] >= start) &
                (train_edge_label_index[0] < end))
        logits[train_edge_label_index[0, mask] - start,
               train_edge_label_index[1, mask] - num_customer] = float('-inf')

        # Computing precision and recall:
        ground_truth = torch.zeros_like(logits, dtype=torch.bool)
        mask = ((data.edge_label_index[0] >= start) &
                (data.edge_label_index[0] < end))
        ground_truth[data.edge_label_index[0, mask] - start,
                     data.edge_label_index[1, mask] - num_customer] = True
        node_count = degree(data.edge_label_index[0, mask] - start,
                            num_nodes=logits.size(0))

        topk_index = logits.topk(k, dim=-1).indices
        isin_mat = ground_truth.gather(1, topk_index)

        precision += float((isin_mat.sum(dim=-1) / k).sum())
        recall += float((isin_mat.sum(dim=-1) / node_count.clamp(1e-6)).sum())
        total_examples += int((node_count > 0).sum())

    return precision / total_examples, recall / total_examples


loss_list = list()
precision_list = list()
recall_list = list()
for epoch in tqdm(range(100)):
    loss = train()
    precision, recall = test(k=20)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Precision@20: '
          f'{precision:.4f}, Recall@20: {recall:.4f}')

    # Collect data for plotting
    loss_list.append(loss)
    precision_list.append(precision)
    recall_list.append(recall)

    if epoch % 10 == 0:
        # Save checkpoint
        torch.save(model, f"../data/LightGCN/checkpoint_{epoch}.pt")

torch.save(model, "../data/LightGCN/model.pt")

# Plot loss
plt.plot(loss_list)
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("../data/LightGCN/loss.png")

plt.cla()

# Plot precision
plt.plot(precision_list)
plt.title("Precision")
plt.xlabel("Epoch")
plt.ylabel("Precision")
plt.savefig("../data/LightGCN/precision.png")

plt.cla()

# Plot recall
plt.plot(recall_list)
plt.title("Recall")
plt.xlabel("Epoch")
plt.ylabel("Recall")
plt.savefig("../data/LightGCN/recall.png")

plt.cla()
