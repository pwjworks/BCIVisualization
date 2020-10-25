import numpy as np
from numpy.lib.index_tricks import IndexExpression
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx

import networkx as nx
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from sklearn.svm import SVC

dataset = Planetoid(root="data/Cora", name="Cora")
# print(dataset.num_classes)
# print(dataset.num_edge_features)

# CoraNet = to_networkx(dataset.data)
# CoraNet = CoraNet.to_undirected()

# Node_degree = pd.DataFrame(data=CoraNet.degree, columns=["Node", "Degree"])
# Node_degree = Node_degree.sort_values(by=["Degree"], ascending=False)
# Node_degree = Node_degree.reset_index(drop=True)
# Node_degree.iloc[0:30, :].plot(
#     x="Node", y="Degree", kind="bar", figsize=(10, 7))
# Node_class = dataset.data.y.data.numpy()

# plt.xlabel("Node", size=12)
# plt.ylabel("Degree", size=12)
# plt.show()

# pos = nx.spring_layout(CoraNet)
# nodecolor = ["red", "blue", "green", "yellow", "peru", "violet", "cyan"]
# nodelabel = np.array(list(CoraNet.nodes))
# plt.figure(figsize=(16, 12))
# for ii in np.arange(len(np.unique(Node_class))):
#     nodelist = nodelabel[Node_class == ii]
#     nx.draw_networkx_nodes(CoraNet, pos, nodelist=list(
#         nodelist), node_size=50, node_color=nodecolor[ii], alpha=0.8)

# nx.draw_networkx_edges(CoraNet, pos, width=1, edge_color="black")
# plt.show()


class GCNnet(torch.nn.Module):
    def __init__(self, input_feature, num_classes):
        super(GCNnet, self).__init__()
        self.input_feature = input_feature
        self.num_classes = num_classes
        self.conv1 = GCNConv(input_feature, 32)
        self.conv2 = GCNConv(32, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.softmax(x, dim=1)


input_feature = dataset.num_node_features
num_classes = dataset.num_classes
mygcn = GCNnet(input_feature, num_classes)

device = torch.device("cuda"if torch.cuda.is_available()else "cpu")
model = mygcn.to(device)
data = dataset[0].to(device)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
train_loss_all = []
val_loss_all = []
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    train_loss_all.append(loss.data.cpu().numpy())
    loss = F.cross_entropy(out[data.val_mask], data.y[data.val_mask])
    val_loss_all.append(loss.data.cpu().numpy())
    if epoch % 20 == 0:
        print("epoch:", epoch, "; Train Loss",
              train_loss_all[-1], "; Val Loss", val_loss_all[-1])

plt.figure(figsize=(10, 6))
plt.plot(train_loss_all, "ro-", label="Train loss")
plt.plot(val_loss_all, "bs-", label="Val Loss")
plt.legend()
plt.grid()
plt.xlabel("epoch", size=13)
plt.ylabel("loss", size=13)
plt.title("Graph Convolutional Networks", size=14)
plt.show()

activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
        return hook


model.conv1.register_forward_hook(get_activation("conv1"))
out = model(data)
conv1 = activation["conv1"].data.cpu().numpy()
print("conv1.shape:", conv1.shape)
