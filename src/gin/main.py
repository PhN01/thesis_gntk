import sys

sys.path.append(".")

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from tqdm import tqdm

from src.gin.util import load_data_tudo, separate_data
from src.gin.models.graphcnn import GraphCNN
from src.config import config as cfg
from src.utils import utils

criterion = nn.CrossEntropyLoss()


def train(config, model, device, train_graphs, optimizer, epoch):
    model.train()

    total_iters = config.iters_per_epoch
    pbar = tqdm(range(total_iters), unit="batch")

    loss_accum = 0
    for pos in pbar:
        selected_idx = np.random.permutation(len(train_graphs))[: config.batch_size]

        batch_graph = [train_graphs[idx] for idx in selected_idx]
        output = model(batch_graph)

        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)

        # compute loss
        loss = criterion(output, labels)

        # backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        # report
        pbar.set_description("epoch: %d" % (epoch))

    average_loss = loss_accum / total_iters
    print("loss training: %f" % (average_loss))

    return average_loss


###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, minibatch_size=64):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i : i + minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)


def test(model, device, train_graphs, test_graphs, epoch):
    model.eval()

    output = pass_data_iteratively(model, train_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in train_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_train = correct / float(len(train_graphs))

    output = pass_data_iteratively(model, test_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))

    print("accuracy train: %f test: %f" % (acc_train, acc_test))

    return acc_train, acc_test, pred


def main():
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(
        description="PyTorch graph convolutional neural net for whole-graph classification"
    )
    parser.add_argument(
        "--dataset", type=str, default="MUTAG", help="name of dataset (default: MUTAG)"
    )
    parser.add_argument(
        "--rep_idx",
        type=int,
        default=0,
        help="the index of the cv iteration. Should be less then 10.",
    )
    parser.add_argument(
        "--fold_idx",
        type=int,
        default=0,
        help="the index of fold in 10-fold validation. Should be less then 10.",
    )
    parser.add_argument(
        "--learn_eps",
        action="store_true",
        help="Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.",
    )
    args = parser.parse_args()

    config = cfg.Config()
    gin_config = cfg.GINConfig(args.dataset)

    seed = 42 + args.rep_idx

    architecture = f"L{gin_config.num_layers}_R{gin_config.num_mlp_layers}_scale{gin_config.neighbor_pooling_type}"
    fold_name = f"rep{args.rep_idx}_fold{args.fold_idx}"

    out_dir = f"{config.exp_path}/GIN/{args.dataset}/{architecture}"
    utils.make_dirs_checked(out_dir)

    # set up seeds and gpu device
    torch.manual_seed(0)
    np.random.seed(0)
    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    graphs, num_classes, g_labels, inv_label_dict = load_data_tudo(args.dataset)

    ##10-fold cross validation. Conduct an experiment on the fold specified by args.fold_idx.
    train_graphs, test_graphs, train_idx, test_idx = separate_data(
        args.dataset, graphs, seed, args.fold_idx, g_labels
    )
    # np.savetxt(f'{out_dir}/{file}_train_indices.txt', train_idx, delimiter=",")
    np.savetxt(f"{out_dir}/{fold_name}_test_indices.txt", test_idx, delimiter=",")

    model = GraphCNN(
        gin_config.num_layers,
        gin_config.num_mlp_layers,
        train_graphs[0].node_features.shape[1],
        gin_config.hidden_dim,
        num_classes,
        gin_config.final_dropout,
        args.learn_eps,
        gin_config.graph_pooling_type,
        gin_config.neighbor_pooling_type,
        device,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=gin_config.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    for epoch in range(1, gin_config.epochs + 1):
        scheduler.step()

        avg_loss = train(gin_config, model, device, train_graphs, optimizer, epoch)
        acc_train, acc_test, _ = test(model, device, train_graphs, test_graphs, epoch)

        with open(f"{out_dir}/{fold_name}.txt", "a") as f:
            f.write("%f %f %f" % (avg_loss, acc_train, acc_test))
            f.write("\n")

        if epoch == gin_config.epochs:
            _, _, predictions = test(model, device, train_graphs, test_graphs, epoch)
            predictions = predictions.data.cpu().numpy().flatten().tolist()
            predictions = [inv_label_dict[pred] for pred in predictions]
            np.savetxt(
                f"{out_dir}/{fold_name}_test_predictions.txt",
                predictions,
                delimiter=",",
            )

        print("")

        print(model.eps)


if __name__ == "__main__":
    main()
