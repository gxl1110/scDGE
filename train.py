import os
import warnings
warnings.filterwarnings('ignore')
import argparse
import torch
from torch.autograd import Variable
import numpy as np
import random
import torch.optim as optim
import torch.nn.functional as F
from cluster_alignment import cluster_alignment_simple
from metric import cluster_accuracy
from losses import pq_loss_func, dist_2_label, sim_loss_func
from utils import weights_init, sparse_to_tuple
from MoeSSL import MoeSSL
import scipy.sparse as sp
from dataload.main import getscData
from vgaeAndGAT import Graph_AE, Discriminator
import time

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda', help='use cpu or gpu.')
parser.add_argument('--gpu', type=int, default=0, help='GPU id.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--dataset', type=str, default='cora', help='Type of dataset.')
parser.add_argument('--encoder', type=str, default="GAT", help='the model name')
parser.add_argument('--hid_dim', type=int, nargs='+', default=[256, 128, 512])
parser.add_argument('--lr_pretrain', type=float, default=1e-4, help='Initial learning rate.')
parser.add_argument('--pretrain_epochs', type=int, default=100, help='Number of pretrained totalssl epochs.')
parser.add_argument('--w_ssl_stage_one', type=float, default=0.25, help='the ssl_loss weight of pretrain stage one')
parser.add_argument('--use_ckpt', type=int, default=0, help='whether to use checkpoint, 0/1.')
parser.add_argument('--save_ckpt', type=int, default=1, help='whether to save checkpoint, 0/1.')

parser.add_argument('--top_k', type=int, default=3, help='The number of experts to choose.')
parser.add_argument('--st_per_p', type=float, default=0.7, help='The threshold of the pseudo positive labels.')
parser.add_argument('--lr_train_fusion', type=float, default=1e-3, help='train pseudo labels learning rate.')
parser.add_argument('--labels_epochs', type=int, default=200, help='Number of epochs to train.')

parser.add_argument('--w_pq', type=float, default=1, help='weight of loss pq.')
parser.add_argument('--w_ssl_stage_two', type=float, default=0.1, help='the ssl_loss weight of train stage')

parser.add_argument('--given_cell_type_labels', action='store_true', default=True,
                    help='(boolean, default False) If True, expect a cell type label file and will compute ARI against those labels')

# Data loading related
parser.add_argument('--load_dataset_dir', type=str, default='./sampleData',
                    help='(str) Folder that stores all your datasets.')
parser.add_argument('--load_dataset_name', type=str, default='Klein',
                    help='(str) Folder that contains all the relevant input files. ')
parser.add_argument('--load_sc_dataset', type=str, default='',
                    help='Not needed if using benchmark')
parser.add_argument('--load_cell_type_labels', type=str, default='',
                    help='Not needed if using benchmark')

parser.add_argument('--load_h5', type=str, default=None,
                    help='(str, default None) If not None, will load the csv generated from the h5 specified in this file path')
parser.add_argument('--load_h5_2', type=str, default=None,
                    help='(str, default None) If not None, will load the csv generated from the h5 specified in this file path')
# Preprocess related
parser.add_argument('--preprocess_cell_cutoff', type=float, default=0.8,
                    help='Not needed if using benchmark')
parser.add_argument('--preprocess_gene_cutoff', type=float, default=0.8,
                    help='Not needed if using benchmark')
parser.add_argument('--preprocess_top_gene_select', type=int, default=2000,
                    help='Not needed if using benchmark')

# Output related
parser.add_argument('--output_dir', type=str, default='./outputs/embedding/',
                    help="(str, default 'outputs/') Folder for storing all the outputs")
parser.add_argument('--output_run_ID', type=int, default=None,
                    help='(int, default None) Run ID to be printed along with metric outputs')
parser.add_argument('--output_preprocessed', action='store_true', default=False,
                    help='(boolean, default False) If true, will output preprocessed data and dropout info')
parser.add_argument('--output_intermediate', action='store_true', default=False,
                    help='(boolean, default False) If true, will output intermediate results')
parser.add_argument('--output_rdata', action='store_true', default=False,
                    help='(boolean, default False) If true, will output rdata results')
parser.add_argument('--n_input', type=int, default=100)

args = parser.parse_args()
opt = vars(args)





def main():

    set_seed(args.seed)
    print(args)

    args.load_dataset_dir = './data'
    args.load_dataset_name = 'Human1'
    args.load_h5_2 = 'Human1.h5'
    args.use_ckpt = 1
    args.lr_pretrain = 1e-4
    args.lr_train_fusion = 1e-2
    args.pretrain_epochs = 100
    args.labels_epochs = 200
    args.save_ckpt = 0

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


    data = getscData(args)

    feat_dim = data.features.shape[1]
    data.labels = np.array(data.labels)
    cluster_num = len(np.unique(data.labels))
    n_nodes = data.features.shape[0]


    encoder = Graph_AE(input_feat_dim=feat_dim)

    modeldis = Discriminator(args.hid_dim[0], args.hid_dim[1], args.hid_dim[2])
    modeldis.apply(weights_init)

    set_of_ssl = ['PairwiseAttrSim', 'PairwiseDistance', 'Par', 'Clu', 'DGI']
    if data.adj.shape[0] > 5000:
        print("use the DGISample")
        local_set_of_ssl = [ssl if ssl != 'DGI' else 'DGISample' for ssl in set_of_ssl]
    else:
        local_set_of_ssl = set_of_ssl
    model = MoeSSL(data, encoder, modeldis, local_set_of_ssl, args, device=args.device).to(args.device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()

    if args.use_ckpt == 1:
        print("load the ssl train model")
        model.encoder.load_state_dict(
            torch.load(
                f'./pretrained/{args.load_dataset_name}_{args.hid_dim[0]}.pkl'))
    else:
        model.train()
        model.TotalSSLpretrain()

    model.encoder.eval()
    _, mu_z, _, _ = model.encoder(model.processed_data.features, model.processed_data.edge_index)

    acc, nmi, f1, ari, mu_pred, kmeans = model.evaluate_pretrained(mu_z)
    print('Z-embddings cluster result: acc:{:.4f}  nmi:{:.4f}  f1:{:.4f} ari:{:.4f} '.format(acc, nmi, f1, ari))

    z_embeddings = mu_z.detach()

    # get the pseudo labels
    cluster_center_z = torch.from_numpy(kmeans.cluster_centers_)
    selected_idx_z, pred_labels_z = generate_pseudo_labels_percentile(z_embeddings, cluster_center_z,
                                                                      data, cluster_num, args.st_per_p)


    params = list(model.gate.parameters())
    # params = []
    for ix, agent in enumerate(model.ssl_agent):
        if agent.disc2 is not None:
            params = params + list(agent.disc2.parameters())
        if hasattr(agent, 'gcn2'):
            params = params + list(agent.gcn2.parameters())

    def get_adj_label(adj):
        sim_matrix = adj.tocsr()
        sim_label = sim_matrix + sp.eye(sim_matrix.shape[0])
        sim_label = sparse_to_tuple(sim_label)
        sim_label = torch.sparse.FloatTensor(torch.LongTensor(sim_label[0].T), torch.FloatTensor(sim_label[1]),
                                             torch.Size(sim_label[2]))
        return sim_label

    adj_label = get_adj_label(data.adj)

    optimizer_train = None
    cluster_centers = None
    # Stage 2: training
    for epoch in range(args.labels_epochs):
        print(epoch)
        model.train()
        fusion_emb, _, _ = model.FeatureFusionForward(z_embeddings)
        if epoch == 0:
            acc, nmi, f1, ari,  moe_pred_labels, kmeans = model.evaluate_pretrained(fusion_emb)
            print('fusion_emb cluster result: acc:{:.4f}  nmi:{:.4f}  f1:{:.4f} ari:{:.4f} '.format(acc, nmi, f1, ari))
            tenor_pseudo_labels = pred_labels_z.unsqueeze(1).cpu()
            pseudo_labels_onehot = torch.zeros(n_nodes, cluster_num).scatter_(1, tenor_pseudo_labels, 1)

            # labels cluster_alignment, return to get the aligned labels1
            align_labels_z = torch.from_numpy(
                cluster_alignment_simple(data, moe_pred_labels, [pseudo_labels_onehot])[0])
            cluster_centers = Variable((torch.from_numpy(kmeans.cluster_centers_).type(torch.FloatTensor)).cuda(),
                                       requires_grad=True)
            params = params + [cluster_centers]
            optimizer_train = optim.Adam(params, lr=args.lr_train_fusion, weight_decay=0.0)

        optimizer_train.zero_grad()

        adj_sim_pred = sample_sim(fusion_emb)
        sim_loss = sim_loss_func(adj_sim_pred.view(-1), adj_label.to(args.device).to_dense().view(-1))

        losspq, p, q = pq_loss_func(fusion_emb, cluster_centers)
        sup_loss = F.nll_loss(torch.log(q[selected_idx_z]), align_labels_z[torch.LongTensor(selected_idx_z)].to(args.device))
        ssl_loss = model.get_ssl_loss_stage_two(z_embeddings, model.processed_data.adj_norm)

        loss = sup_loss + sim_loss + args.w_pq * losspq + args.w_ssl_stage_two * ssl_loss
        print(
            "loss pq: {}, pos_loss: {}, ssl_loss: {}, sim_loss: {}".format(losspq, sup_loss, ssl_loss, sim_loss,))

        loss.backward()
        optimizer_train.step()

        model.eval()
        fusion_emb, _, _ = model.FeatureFusionForward(z_embeddings)
        losspq, p, q = pq_loss_func(fusion_emb, cluster_centers)
        cluster_pred_score, cluster_pred = dist_2_label(q)
        acc, nmi, f1, ari = cluster_accuracy(cluster_pred.cpu(), data.labels, cluster_num)
        print('Evaluate encoder_decoder cluster result: acc:{:.4f}  nmi:{:.4f}  f1:{:.4f} ari:{:.4f} '.format(acc, nmi, f1, ari))

    print("training ending...")



    model.eval()
    fusion_emb, _, nodes_weight = model.FeatureFusionForward(z_embeddings)
    _, p, q = pq_loss_func(fusion_emb, cluster_centers)
    _, cluster_pred = dist_2_label(q)
    acc, nmi, f1, ari = cluster_accuracy(cluster_pred.cpu(), data.labels, cluster_num)


    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()

    times = end - start

    torch.cuda.synchronize()

    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    csv_filename = "./results/scDGE_results.csv"

    save_results_to_csv_time(csv_filename, args.load_dataset_name, acc, nmi, ari, times, peak_alloc, peak_reserved)


def init():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    print("-----------------------------------------------------------------")
    print("-----------------------------------------------------------------")
    print("-----------------------------------------------------------------")



# pred adj
def sample_sim(fusion_emb, xind=None, yind=None):

    def scale(z):
        zmax = z.max(dim=1, keepdim=True)[0]
        zmin = z.min(dim=1, keepdim=True)[0]
        z_std = (z - zmin) / (zmax - zmin)
        z_scaled = z_std
        return z_scaled
    fusion_scaled = scale(fusion_emb)
    fusion_norm = F.normalize(fusion_scaled)
    sim_adj = torch.mm(fusion_norm, fusion_norm.t())

    return sim_adj


def set_seed(seed):
    SEED = seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

if __name__ == '__main__':
    init()
    main()
