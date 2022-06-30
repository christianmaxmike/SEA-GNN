# __author__ = "Christian Frey"
from ogb.graphproppred import Evaluator
from sklearn.metrics import confusion_matrix
import torch
import numpy as np
from tqdm import tqdm

def accuracy_SBM(scores, targets):
    S = targets.cpu().numpy()
    C = np.argmax(torch.nn.Softmax(dim=1)(scores).cpu().detach().numpy(), axis=1)
    CM = confusion_matrix(S, C).astype(np.float32)
    nb_classes = CM.shape[0]
    targets = targets.cpu().detach().numpy()
    nb_non_empty_classes = 0
    pr_classes = np.zeros(nb_classes)
    for r in range(nb_classes):
        cluster = np.where(targets == r)[0]
        if cluster.shape[0] != 0:
            pr_classes[r] = CM[r, r] / float(cluster.shape[0])
            if CM[r, r] > 0:
                nb_non_empty_classes += 1
        else:
            pr_classes[r] = 0.0
    acc = 100. * np.sum(pr_classes) / float(nb_classes)
    return acc


def evaluate(model, dataloader, device, params):
    model.eval()
    if "ogb" in params['dataset']:
        evaluator = Evaluator(name=params['dataset'])

    pbar = tqdm(total=len(dataloader))
    targets = []
    scores = []
    total_loss = 0.0
    result_dict = {}
    test_acc = 0.0
    expert_count = torch.zeros(model.num_experts)

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
            test_acc += accuracy_SBM(y_pred, y_batch)

        if experts_cnt is not None:
            cnt = torch.bincount(experts_cnt.cpu(), minlength=model.num_experts)
            expert_count += cnt

        total_loss += loss.item()
        targets.append(y_batch.detach().cpu().numpy())
        scores.append(y_pred.detach().cpu().numpy())
        pbar.update(1)
    pbar.close()
    targets = np.concatenate(targets, axis=0)
    scores = np.concatenate(scores, axis=0)
    if "ogb" in params['dataset']:
        input_dict = {"y_true": targets, "y_pred": scores}
        result_dict = evaluator.eval(input_dict)

    print("Expert cnt:", expert_count / sum(expert_count))
    return result_dict, total_loss / len(dataloader), test_acc / len(dataloader), expert_count/sum(expert_count)
