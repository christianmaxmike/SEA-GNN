import argparse
import json
import math
import sys
import traceback

import numpy as np
import os
import pickle
import time
from tqdm import tqdm
import random

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from DataLoader import MolHIVDataset, MoleculeDatasetDGL, SBMsDatasetDGL
import SEAGNN, SEAAGG, TBase
import utils
import evaluate
import matplotlib.pyplot as plt


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def extract_embed(node_embed, input_nodes, eps=1e-12):
    emb = {}
    for ntype, nid in input_nodes.items():
        nid = input_nodes[ntype]
        res = node_embed[ntype][nid].clone()
        res[node_embed[ntype][nid] == 0] = eps
        emb[ntype] = res
    return emb


def adjust_learning_rate(optimizer, epoch, args):
    """Learning rate scheduler"""
    lr = args.lr
    if args.cos_lr:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.n_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def parse_args():
    parser = argparse.ArgumentParser(description="KGTransformer")

    parser.add_argument("--config", type=str, default="zinc_seagnn", help="json file including configs")
    parser.add_argument("--early_stop", type=int, default=25, help="max testing for early stopping")
    parser.add_argument("--model_path", type=str, default="checkpts", help='path for save the model')
    parser.add_argument("--model_suffix", type=str, default="_test")
    parser.add_argument("--load", type=str, default=None)

    # READ ARGS
    args = parser.parse_args()
    # args.model_folder = os.path.join(args.model_path, "model_{}{}".format(args.dataset, args.model_suffix))
    # if not os.path.exists(os.path.join(args.model_folder)):
    #     os.makedirs(os.path.join(args.model_folder))
    # with open(os.path.join(args.model_folder, "config"), "wb") as handle:
    #     pickle.dump(args, handle)

    return args


def save_model(model, train_exp_dist, valid_exp_dist, test_exp_dist, params, model_suffix):
    torch.save(model.state_dict(), 'model_best_{}_{}_{}.pth'.format(params['dataset'], params['model'], model_suffix))

    for type,file in zip(['train', 'valid', 'test'], [train_exp_dist, valid_exp_dist, test_exp_dist]):
        with open("./{}_expdist_best_{}_{}_{}.pt".format(type, params['dataset'], params['model'], model_suffix), "wb") as handle:
            pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(model, load_path):
    model.load_state_dict(torch.load(load_path))


def store_recordings(train_losses, valid_losses, test_losses,
                     train_aucs, valid_aucs, test_aucs,
                     params, model_suffix):
    types = ['train', 'valid', 'test']
    for type, file in zip(types, [train_losses, valid_losses, test_losses]):
        with open("./{}_losses_best_{}_{}_{}.pt".format(type, params['dataset'], params['model'], model_suffix), "wb") as handle:
            pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)
    for type, file in zip(types, [train_aucs, valid_aucs, test_aucs]):
        with open("./{}_aucs_best_{}_{}_{}.pt".format(type, params['dataset'], params['model'], model_suffix), "wb") as handle:
            pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)


def process(model, optimizer, scheduler, device, params, train_loader, test_loader, valid_loader, args):
    c_early_stop = 0
    time_avgemeter = utils.AverageMeter("EpochTime", ":.2f")

    train_losses, train_aucs = [], []
    valid_losses, valid_aucs = [], []
    test_losses, test_aucs = [], []
    best_exp_cnt = None
    if "ogb" in params['dataset'] or 'pattern' == params['dataset']:
        best_test_auc = -1
    else:
        best_test_auc = sys.maxsize

    for epoch in range(params['epochs']):
        start_epoch = time.time()
        epoch_loss = training(model, optimizer, train_loader, device, params)
        time_avgemeter.update(time.time() - start_epoch)
        print("Epoch[{}/{}]; Epoch train loss:{:.4f}; {}".format(epoch, params["epochs"], epoch_loss, time_avgemeter))

        if (epoch + 1) % params['eval_freq'] == 0:
            c_early_stop += 1

            train_dict, train_loss, train_acc, train_exp_cnt = evaluate.evaluate(model, train_loader, device, params)
            valid_dict, valid_loss, valid_acc, valid_exp_cnt = evaluate.evaluate(model, valid_loader, device, params)
            test_dict, test_loss, test_acc, test_exp_cnt = evaluate.evaluate(model, test_loader, device, params)
            print("Train: ", train_dict, "\tTrain Loss: ", train_loss, "\tTrain acc: ", train_acc)
            print("Valid: ", valid_dict, "\tValid Loss: ", valid_loss, "\tValid acc: ", valid_acc)
            print("Test: ", test_dict, "\tTest Loss: ", test_loss, "\tTest acc: ", test_acc)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            test_losses.append(test_loss)

            if "ogb" in params['dataset']:
                train_aucs.append(train_dict[list(train_dict.keys())[0]])
                valid_aucs.append(valid_dict[list(train_dict.keys())[0]])
                test_aucs.append(test_dict[list(train_dict.keys())[0]])
                if best_test_auc < test_dict[list(train_dict.keys())[0]]:
                    best_test_auc = test_dict[list(train_dict.keys())[0]]
                    best_exp_cnt = test_exp_cnt
                    c_early_stop = 0
                    save_model(model, train_exp_cnt, valid_exp_cnt, test_exp_cnt, params, args.model_suffix)
            elif params['dataset'] == 'zinc':
                if best_test_auc > test_loss:
                    best_test_auc = test_loss
                    best_exp_cnt = test_exp_cnt
                    c_early_stop = 0
                    save_model(model, train_exp_cnt, valid_exp_cnt, test_exp_cnt, params, args.model_suffix)
            else:
                train_aucs.append(train_acc)
                valid_aucs.append(valid_acc)
                test_aucs.append(test_acc)
                if best_test_auc < test_acc:
                    best_test_auc = test_acc
                    best_exp_cnt = test_exp_cnt
                    c_early_stop = 0
                    save_model(model, train_exp_cnt, valid_exp_cnt, test_exp_cnt, params, args.model_suffix)

            store_recordings(train_losses, valid_losses, test_losses,
                             train_aucs, valid_aucs, test_aucs,
                             params, args.model_suffix)
            scheduler.step(valid_loss)

        if optimizer.param_groups[0]['lr'] < params['min_lr']:
            print("\nLearning rate reached min learning rate.")
            break

        if c_early_stop == args.early_stop:
            print("\nEarly stopping. No more improvements.")
            break
    print("Best result: {}".format(best_test_auc))
    print("Expert dist: ", ["{:.2f}".format(i.item()) for i in best_exp_cnt])


def training(model, optimizer, dataloader, device, params, verbose_epochs=10):
    model.train()
    start = time.time()
    train_loss = 0.0
    expert_count = torch.zeros(model.num_experts)

    pbar = tqdm(total=len(dataloader))
    for i, (x_batch, y_batch) in enumerate(dataloader):
        x_batch = x_batch.to(device)
        batch_n = x_batch.ndata['feat'].to(device)
        batch_e = x_batch.edata['feat'].to(device)
        y_batch = y_batch.to(device)

        try:
            batch_lap_pos_enc = x_batch.ndata['lap_pos_enc'].to(device)
            sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            batch_lap_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0)
        except:
            batch_lap_pos_enc = None

        if params['dataset'] == "ogbg-molhiv":
            y_pred, experts_cnt = model(x_batch, batch_n, batch_e, True, True, batch_lap_pos_enc)
            loss = model.loss_fnc(params['loss_fnc'], y_pred, y_batch)
        elif params['dataset'] == "ogbg-molpcba":
            y_pred = model(x_batch, batch_n, batch_e, True, True, batch_lap_pos_enc)
            is_labeled = y_batch == y_batch
            loss = model.loss_fnc(params['loss_fnc'], y_pred[is_labeled], y_batch[is_labeled])
        elif params['dataset'] == "zinc":
            y_pred, experts_cnt = model(x_batch, batch_n, batch_e, False, True, batch_lap_pos_enc)
            loss = model.loss_fnc(params['loss_fnc'], y_pred, y_batch)
        elif params['dataset'] == "pattern":
            y_pred, experts_cnt = model(x_batch, batch_n, batch_e, False, False, batch_lap_pos_enc)
            loss = model.loss_fnc(params['loss_fnc'], y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
        if i % 10 == 0:
            plot_grad_flow(model.named_parameters(), filename="gradients_{}.png".format(i))
        optimizer.step()

        train_loss += loss.item()

        if experts_cnt is not None:
            cnt = torch.bincount(experts_cnt.cpu(), minlength=model.num_experts)
            expert_count += cnt

        pbar.update(1)
        if (i+1) % verbose_epochs == 0:
            loss_avg = train_loss / (i+1)
            pbar.set_postfix({'loss': "{:.4f}".format(loss_avg),
                              'time': "{:.4f}".format(time.time() - start)})
            start = time.time()

    pbar.close()

    return train_loss/len(dataloader)


def init_optimizer(args, model_params):
    if args["type"] == "adam":
        optim = torch.optim.Adam(model_params, lr=args["lr"],
                                 weight_decay=args["weight_decay"], betas=(args["beta1"], args["beta2"]))
    elif args["type"] == "adamw":
        optim = torch.optim.AdamW(model_params, lr=args['lr'],
                                  weight_decay=args["weight_decay"], betas=(args["beta1"], args["beta2"]))
    elif args["type"] == "adagrad":
        optim = torch.optim.Adagrad(model_params, lr=args["lr"],
                                    weight_decay=args["weight_decay"])
    elif args["type"] == "sgd":
        optim = torch.optim.SGD(model_params, lr=args["lr"],
                                weight_decay=args["weight_decay"], momentum=args["momentum"])
    else:
        raise NotImplementedError("Unknown optimizer: {}".format(args["type"]))
    return optim


def load_data(model, name, params):
    """ Load dataset """
    if name == "ogbg-molhiv":
        dataset = MolHIVDataset()
        if params['k_hop']:
            dataset.addkhops(params['k_cutoff'])
        if params['lap_pos_enc']:
            st = time.time()
            print("[!] Adding Laplacian positional encoding.")
            dataset._add_laplacian_positional_encodings(params['pos_enc_dim'])
            print('Time LapPE:', time.time() - st)
    elif name == "zinc":
        dataset = MoleculeDatasetDGL(ppr_flag=True if model == "SEAPPR" else False)
        if params['full_graph']:
            dataset._make_full_graph()
        if params['k_hop']:
            dataset._addkhops(params['k_cutoff'])
        if params['lap_pos_enc']:
           st = time.time()
           print("[!] Adding Laplacian positional encoding.")
           dataset._add_laplacian_positional_encodings(params['pos_enc_dim'])
           print('Time LapPE:', time.time() - st)
        if params['pr_enc']:
            st = time.time()
            print("Adding PageRank info encodings.")
            dataset._add_ppr_encodings(params['ppr_alphas'])
            print('PTimePageRank:', time.time() - st)

        # add k-hop graphs
        # rdataset._add_kneighorhood_graphs(params['num_experts'])

    elif name == "pattern":
        dataset = SBMsDatasetDGL('PATTERN')
        if params['k_hop']:
            dataset._addkhops(params['k_cutoff'])
        if params['lap_pos_enc']:
            st = time.time()
            print("[!] Adding Laplacian positional encoding.")
            dataset._add_laplacian_positional_encodings(params['pos_enc_dim'])
            print('Time LapPE:', time.time() - st)
    else:
        raise NotImplementedError("Dataset not found: {}".format(name))
    return dataset


class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


def plot_grad_flow(named_parameters, filename="gradients.png"):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n) and (p.grad is not None):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig(filename, bbox_inches="tight")

def main():
    args = parse_args()
    with open(os.path.join("configs", args.config)) as f:
        params = json.load(f)

    device = utils.check_cuda(params['gpu'], params['gpu_id'])
    dataset = load_data(params['model'], params['dataset'], params['net_params'])
    train_loader = DataLoader(dataset.train, batch_size=params['batch_size'],
                              shuffle=True, collate_fn=dataset.collate_dgl, num_workers=0)
    test_loader = DataLoader(dataset.test, batch_size=params['batch_size'],
                             shuffle=True, collate_fn=dataset.collate_dgl, num_workers=0)
    valid_loader = DataLoader(dataset.valid, batch_size=params['batch_size'],
                              shuffle=True, collate_fn=dataset.collate_dgl, num_workers=0)

    try:
        n_entities = None
        n_rels = None
        if 'ogb' not in params["dataset"]:
            n_entities = dataset.num_entities
            n_rels = dataset.num_rels
        model = globals()[params['model']].Encoder(params["net_params"], n_entities, n_rels).to(device)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Model/Total params: ", pytorch_total_params)
    except KeyError:
        print("Error while initializing the model: {}".format(params['model']))
        traceback.print_exc()
        exit(-1)

    optimizer = init_optimizer(params['optimizer'], model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=params['lr_reduce_factor'],
                                                           patience=params['lr_schedule_patience'],
                                                           min_lr=params['min_lr'],
                                                           verbose=True)

    # setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])

    process(model, optimizer, scheduler, device, params, train_loader, test_loader, valid_loader, args)


if __name__ == '__main__':
    main()
