from critical_look_utils.dataset import Dataset
from critical_look_utils.utils import Logger, get_parameter_groups, get_lr_scheduler_with_warmup
from critical_look_utils.model_GCNw import Model
import torch
from collections import defaultdict
from torch_geometric.utils import to_dense_adj, remove_self_loops, dense_to_sparse
from utils.helper import get_eig
from utils.data_loading import build_edge_index
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import argparse
import os
import yaml
import dgl
from dgl import ops
import scipy.sparse as sp
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=None, help='Experiment name. If None, model name is used.')
    parser.add_argument('--save_dir', type=str, default='new_heter_GCNw', help='Base directory for saving information.')
    parser.add_argument('--dataset', type=str, default='minesweeper',
                        choices=['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions',
                                 'squirrel', 'squirrel-directed', 'squirrel-filtered', 'squirrel-filtered-directed',
                                 'chameleon', 'chameleon-directed', 'chameleon-filtered', 'chameleon-filtered-directed',
                                 'actor', 'texas', 'texas-4-classes', 'cornell', 'wisconsin'])
    # model architecture
    parser.add_argument('--model', type=str, default='GCNw')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--hidden_dim_multiplier', type=float, default=1)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--normalization', type=str, default='LayerNorm', choices=['None', 'LayerNorm', 'BatchNorm'])
    # regularization
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=0)
    # training parameters
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--num_warmup_steps', type=int, default=None,
                        help='If None, warmup_proportion is used instead.')
    parser.add_argument('--warmup_proportion', type=float, default=0, help='Only used if num_warmup_steps is None.')
    # node feature augmentation
    parser.add_argument('--use_sgc_features', default=False, action='store_true')
    parser.add_argument('--use_identity_features', default=False, action='store_true')
    parser.add_argument('--use_adjacency_features', default=False, action='store_true')
    parser.add_argument('--do_not_use_original_features', default=False, action='store_true')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_runs', type=int, default=10)
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--amp', default=False, action='store_true')
    # dgat parameter
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
    parser.add_argument('--alpha', type=float, default=0.9, help='alpha')
    args = parser.parse_args()
    if args.name is None:
        args.name = args.model
    return args

def train_step(model, aug_W, dataset, optimizer, scheduler, scaler, amp=False):
    model.train()
    with autocast(enabled=amp):
      logits = model(adj=aug_W, x=dataset.node_features)
      loss = dataset.loss_fn(input=logits[dataset.train_idx], target=dataset.labels[dataset.train_idx])
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    scheduler.step()

@torch.no_grad()
def evaluate(model, aug_W, dataset, amp=False):
    model.eval()
    with autocast(enabled=amp):
      logits = model(adj=aug_W, x=dataset.node_features)
    metrics = dataset.compute_metrics(logits)
    return metrics

def double_aug_rw(adj, gamma, alpha):
    # l_norm = (gamma * D + (1-gamma) * I)^-alpha
    # middle = (gamma * A + (1-gamma) * I) 
    # r_norm = (gamma * D + (1-gamma) * I)^(alpha - 1)
    N = adj.shape[0]
    adj.setdiag(0) # remove any self-loop
    rowsum = np.array(adj.sum(1)) + 1e-20
    aug_rowsum = gamma * rowsum + (1-gamma) * np.ones((N, 1))
    l_norm = np.power(aug_rowsum, -alpha).flatten()
    l_norm[np.isinf(l_norm)] = 0.
    l_norm = sp.diags(l_norm, 0)
    r_norm = np.power(aug_rowsum, alpha-1).flatten()
    r_norm[np.isinf(r_norm)] = 0.
    r_norm = sp.diags(r_norm, 0)
    middle = gamma * adj + (1-gamma) * sp.eye(N)
    aug_W = l_norm @ middle @ r_norm
    return aug_W

torch.manual_seed(0)
args = get_args()
dataset = Dataset(name=args.dataset,
                add_self_loops=False,
                device=args.device,
                use_sgc_features=args.use_sgc_features,
                use_identity_features=args.use_identity_features,
                use_adjacency_features=args.use_adjacency_features,
                do_not_use_original_features=args.do_not_use_original_features)

# dgat prepare
graph = dataset.graph
edge_index = graph.adj().coalesce().indices()
nnodes = graph.number_of_nodes()
adj = to_dense_adj(edge_index.cpu().to(dtype=torch.int64))[0].numpy()
# direction_info = get_eig(adj, norm=None, gamma=args.gamma, alpha=args.alpha)
adj = sp.csr_matrix(adj)
# augmented adjacency matrix
aug_W = double_aug_rw(adj, args.gamma, args.alpha)
aug_W = aug_W.toarray()
aug_W = torch.tensor(aug_W).to(dtype=torch.float32).to(args.device)

#for layers in range(1, 6):
#args.num_layers = layers
logger = Logger(args, metric=dataset.metric, num_data_splits=dataset.num_data_splits)
for run in range(1, args.num_runs + 1):
    model = Model(model_name=args.model,
                    num_layers=args.num_layers,
                    input_dim=dataset.num_node_features,
                    hidden_dim=args.hidden_dim,
                    output_dim=dataset.num_targets,
                    hidden_dim_multiplier=args.hidden_dim_multiplier,
                    num_heads=args.num_heads,
                    normalization=args.normalization,
                    dropout=args.dropout).to(args.device)
    parameter_groups = get_parameter_groups(model)
    optimizer = torch.optim.AdamW(parameter_groups, lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp)
    scheduler = get_lr_scheduler_with_warmup(optimizer=optimizer, num_warmup_steps=args.num_warmup_steps,
                                                num_steps=args.num_steps, warmup_proportion=args.warmup_proportion)
    logger.start_run(run=run, data_split=dataset.cur_data_split + 1)
    with tqdm(total=args.num_steps, desc=f'Run {run}', disable=args.verbose) as progress_bar:
        for step in range(1, args.num_steps + 1):
            train_step(model=model, aug_W=aug_W, dataset=dataset, optimizer=optimizer, scheduler=scheduler,
                        scaler=scaler, amp=args.amp)
            metrics = evaluate(model=model, aug_W=aug_W, dataset=dataset, amp=args.amp)
            logger.update_metrics(metrics=metrics, step=step)
            progress_bar.update()
            progress_bar.set_postfix({metric: f'{value:.2f}' for metric, value in metrics.items()})
    logger.finish_run()
    model.cpu()
    dataset.next_data_split()
logger.print_metrics_summary()
with open(os.path.join(logger.save_dir, 'metrics.yaml'), 'r') as file:
    metrics = yaml.safe_load(file)
test_mean = metrics[f"test {logger.metric} mean"]
test_std = metrics[f"test {logger.metric} std"]
filename = f'new_heter_GCNw/{args.dataset}_norewire.csv'
print(f"Saving results to {filename}")
with open(f"{filename}", 'a+') as write_obj:
    write_obj.write(f"{args.model.lower()}, " +
                    f"{test_mean:.4f}, " +
                    f"{test_std:.4f}, " +
                    f"{args.gamma}, "+
                    f"{args.alpha}, "+
                    f"{args.num_layers}, "+
                    f"{args}\n")