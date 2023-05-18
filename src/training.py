import os
import os.path as osp

import torch
import poptorch
import pandas as pd
import py3Dmol

from periodictable import elements
from poptorch_geometric.dataloader import CustomFixedSizeDataLoader

from torch_geometric.datasets import QM9
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import to_fixed_size
from torch_geometric.nn.models import SchNet
from torch_geometric.transforms import Pad
from torch_geometric.transforms.pad import AttrNamePadding
from tqdm import tqdm

from utils import TrainingModule, KNNInteractionGraph, prepare_data, optimize_popart

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

poptorch.setLogLevel("ERR")
executable_cache_dir = (
    os.getenv("POPLAR_EXECUTABLE_CACHE_DIR", "/tmp/exe_cache/") + "/pyg-schnet"
)
dataset_directory = os.getenv("DATASETS_DIR", "data")
num_ipus = os.getenv("NUM_AVAILABLE_IPU", "4")

# download qm9 dataset
qm9_root = osp.join(dataset_directory, "qm9")
dataset = QM9(qm9_root)

loader = DataLoader(dataset, batch_size=4)

it = iter(loader)
next(it), next(it)

# run model on cpu
batch = Batch.from_data_list([dataset[0]])
torch.manual_seed(0)
cutoff = 10.0
model = SchNet(cutoff=cutoff)
model.eval()
cpu = model(batch.z, batch.pos, batch.batch)

torch.manual_seed(0)
knn_graph = KNNInteractionGraph(cutoff=cutoff, k=batch.num_nodes - 1)
model = SchNet(cutoff=cutoff, interaction_graph=knn_graph)
model = to_fixed_size(model, batch_size=1)
options = poptorch.Options()
options.enableExecutableCaching(executable_cache_dir)
pop_model = poptorch.inferenceModel(model, options)
ipu = pop_model(batch.z, batch.pos, batch.batch)

assert torch.allclose(cpu, ipu)
