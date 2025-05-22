import os
import re
import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
import shap
import matplotlib.pyplot as plt
import mne
from matplotlib.colors import LinearSegmentedColormap
from mne.viz import plot_topomap
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from imports import *
from models.additional import (
    all_participants_data,   
    create_token_groups,
    create_torch_dataset,
    prepare_data_for_pytorch
)
from models.mlp import MLPModel
from models.cnn import MultiHeadConv1DModel
from models.transformer import FeatureGroupTransformerModel



channel_mapping = {
     1: 'FT7',  2: 'FT8',  3: 'T7',  4: 'T8',  5: 'TP7',  6: 'TP8',
     7: 'CP1',  8: 'CP2',  9: 'P1', 10: 'Pz', 11: 'P2', 12: 'PO3',
    13: 'POz', 14: 'PO4', 15: 'O1', 16: 'Oz', 17: 'O2'
}

montage = mne.channels.make_standard_montage('standard_1020')
pos_dict = montage.get_positions()['ch_pos']

common_cmap = plt.cm.jet

def plot_feature_importances(names, importances, top_n=20, title='Feature Importances'):
    items = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)[:top_n]
    feats, vals = zip(*items)
    plt.figure(figsize=(10,8))
    plt.barh(feats[::-1], vals[::-1])
    plt.title(title)
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()

def compute_model_importances(
    all_data,
    model_type='transformer',
    task_type='binary',
    batch_size=32,
    learning_rate=1e-4,
    quick_train_epochs=1,
    shap_nsamples=200,
    subject_calibration=0.0
):
    X_flat, y_bin, y_ter, y_cont, groups, feature_names = prepare_data_for_pytorch(all_data)
    if task_type=='binary':
        y, loss_fn = y_bin, nn.BCELoss()
    elif task_type=='ternary':
        y, loss_fn = y_ter, nn.CrossEntropyLoss()
    else:
        y, loss_fn = y_cont, nn.MSELoss()
    feature_groups = create_token_groups(all_data)
    gkf = GroupKFold(n_splits=len(np.unique(groups)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def mlp_sampling(m, te_dl, bg_dl):
        m_cpu = m.cpu().eval()
        Xbg = next(iter(bg_dl))[0].cpu().float()
        Xts = next(iter(te_dl))[0].cpu().float()
        explainer = shap.DeepExplainer(m_cpu, Xbg)
        shap_vals = explainer.shap_values(Xts)
        if isinstance(shap_vals, (list, tuple)):
            arr = np.vstack([v if isinstance(v, np.ndarray) else v.cpu().numpy() for v in shap_vals])
        else:
            arr = shap_vals if isinstance(shap_vals, np.ndarray) else shap_vals.cpu().numpy()
        return arr, np.mean(np.abs(arr), axis=0), Xts.cpu().numpy()

    def cnn_grad(m, te_dl, bg_dl):
        Xts, _ = next(iter(te_dl))
        Xts = Xts.to(device).requires_grad_()
        (m(Xts).sum()).backward()
        grads = Xts.grad.abs().cpu().numpy()
        return grads, grads.mean(axis=0)

    def transformer_attn(m, te_dl, bg_dl):
        m_cpu = m.cpu().eval()
        Xts, _ = next(iter(te_dl))
        with torch.no_grad():
            _, attn = m_cpu(Xts.cpu(), return_attention_weights=True)
        grp = attn.mean(dim=0).numpy()
        imp = np.zeros(len(feature_names))
        for i, (_, (st, sz)) in enumerate(feature_groups.items()):
            imp[st:st+sz] = grp[i] / sz
        return attn.cpu().numpy(), imp

    imp_fn = {'mlp': mlp_sampling, 'cnn': cnn_grad}.get(model_type, transformer_attn)
    all_feats, extra_info = [], None

    for train_idx, test_idx in tqdm(gkf.split(X_flat, y, groups), desc="CV folds", total=len(np.unique(groups))):
        if subject_calibration>0:
            ncal = int(len(test_idx)*subject_calibration)
            if ncal>0:
                cal = np.random.choice(test_idx, ncal, replace=False)
                train_idx = np.concatenate([train_idx, cal])
                test_idx = np.setdiff1d(test_idx, cal)
        Xtr, Xte = X_flat[train_idx], X_flat[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        scaler = StandardScaler().fit(Xtr)
        Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)
        tr_dl = DataLoader(create_torch_dataset(Xtr, ytr, task_type), batch_size=batch_size, shuffle=True)
        te_dl = DataLoader(create_torch_dataset(Xte, yte, task_type), batch_size=batch_size, shuffle=False)

        out_dim = 3 if task_type=='ternary' else 1
        model = {
            'mlp': MLPModel(input_size=Xtr.shape[1], output_size=out_dim, task_type=task_type),
            'cnn': MultiHeadConv1DModel(feature_groups, output_size=out_dim, task_type=task_type),
        }.get(model_type, FeatureGroupTransformerModel(feature_groups, output_size=out_dim, task_type=task_type))
        model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

        model.train()
        for _ in range(quick_train_epochs):
            for xb, yb in tr_dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()

        result = imp_fn(model, te_dl, tr_dl)
        if model_type=='mlp':
            shap_vals, feat_imp, Xts = result
            extra_info = (shap_vals, Xts)
        elif model_type=='cnn':
            grads, feat_imp = result
            extra_info = grads
        else:
            attn, feat_imp = result
            extra_info = attn

        all_feats.append(feat_imp)

    feat_imp = np.mean(np.stack(all_feats), axis=0)
    group_imp = {nm: feat_imp[st:st+sz].mean() for nm, (st, sz) in feature_groups.items()}
    return feat_imp, group_imp, extra_info, feature_names

def compute_channel_importances(feat_imp, feature_names):
    imp5, imp2 = {}, {}
    for ch, ch_lbl in channel_mapping.items():
        idx5 = [i for i, n in enumerate(feature_names) if n.startswith(f'EEG_5Bands_{ch_lbl}_')]
        imp5[ch_lbl] = feat_imp[idx5].mean() if idx5 else 0.0
        idx2 = [i for i, n in enumerate(feature_names) if n.startswith(f'EEG_2Hz_{ch_lbl}_')]
        imp2[ch_lbl] = feat_imp[idx2].mean() if idx2 else 0.0
    return imp5, imp2

def plot_channel_importance_topomap(importances, title=None, cmap=common_cmap):
    names = list(importances.keys())
    data = np.array(list(importances.values()))
    pos = np.array([pos_dict[n][:2] for n in names])
    fig, ax = plt.subplots(figsize=(6,6), subplot_kw={'aspect':'equal'})
    im, _ = plot_topomap(data, pos, axes=ax, show=False, names=names, cmap=cmap, extrapolate='local')
    if title:
        ax.set_title(title)
    ax.axis('off')
    cax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('Importance')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()
    return fig

if __name__=='__main__':
    model_type = 'transformer'  
    feat_imp, group_imp, info, feature_names = compute_model_importances(
        all_data=all_participants_data,
        model_type=model_type,
        task_type='binary',
        batch_size=128,
        learning_rate=1e-4,
        quick_train_epochs=2,
        shap_nsamples=100,
        subject_calibration=0.0
    )

    feature_groups = create_token_groups(all_participants_data)


    renamed_feature_names = []
    for name in feature_names:
        m = re.search(r'Channel_(\d+)', name)
        if m:
            num = int(m.group(1))
            lbl = channel_mapping[num]
            renamed_feature_names.append(name.replace(f'Channel_{num}', lbl))
        else:
            renamed_feature_names.append(name)
    feature_names = renamed_feature_names

    renamed_feature_groups = {}
    for nm, (st, sz) in feature_groups.items():
        m = re.search(r'Channel_(\d+)', nm)
        if m:
            num = int(m.group(1))
            lbl = channel_mapping[num]
            new_nm = nm.replace(f'Channel_{num}', lbl)
        else:
            new_nm = nm
        renamed_feature_groups[new_nm] = (st, sz)
    feature_groups = renamed_feature_groups

    if model_type=='mlp':
        shap_vals, Xts = info
        group_shap = np.stack([
            shap_vals[:, st:st+sz].mean(axis=1)
            for _, (st, sz) in feature_groups.items()
        ], axis=1)
        shap.summary_plot(group_shap, group_shap, feature_names=list(feature_groups.keys()), plot_type="violin")
        names, vals = zip(*sorted(group_imp.items(), key=lambda x: -x[1]))
        plt.figure(figsize=(12,9))
        plt.barh(names[::-1], vals[::-1])
        plt.title("Top Groups")
        plt.tight_layout()
        plt.show()
        imp5, imp2 = compute_channel_importances(feat_imp, feature_names)
        imp_combined = {ch: (imp5[ch] + imp2[ch]) / 2 for ch in imp5}
        plot_channel_importance_topomap(imp_combined, 'MLP EEG Topomap')

    elif model_type=='cnn':
        grads = info
        plt.figure(figsize=(24,4))
        ax = sns.heatmap(grads[0:1], cbar=True, cmap=common_cmap)
        ticks, labels = [], []
        for name, (st, sz) in feature_groups.items():
            ticks.append(st + sz//2)
            labels.append(name)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticks([0])
        ax.set_yticklabels(['grad'])
        plt.title("CNN per-group gradient magnitude")
        plt.tight_layout()
        plt.show()
        names, vals = zip(*sorted(group_imp.items(), key=lambda x: -x[1]))
        plt.figure(figsize=(12,9))
        plt.barh(names[::-1], vals[::-1])
        plt.title("Top Groups")
        plt.tight_layout()
        plt.show()
        imp5, imp2 = compute_channel_importances(feat_imp, feature_names)
        imp_combined = {ch: (imp5[ch] + imp2[ch]) / 2 for ch in imp5}
        plot_channel_importance_topomap(imp_combined, 'CNN EEG Topomap')

    else:
        attn = info
        corr = np.corrcoef(attn, rowvar=False)
        plt.figure(figsize=(16,12))
        sns.heatmap(
            corr,
            xticklabels=list(feature_groups.keys()),
            yticklabels=list(feature_groups.keys()),
            cmap=common_cmap,
            center=0
        )
        plt.title("Transformer Attention Correlation Matrix")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

        group_att = attn.mean(axis=0)
        names, vals = zip(*sorted(zip(feature_groups.keys(), group_att), key=lambda x: -x[1]))
        plt.figure(figsize=(12,9))
        plt.barh(names[::-1], vals[::-1])
        plt.title("Top Attention Groups")
        plt.tight_layout()
        plt.show()

        feat_att = np.zeros(len(feature_names))
        for i, (_, (st, sz)) in enumerate(feature_groups.items()):
            feat_att[st:st+sz] = group_att[i] / sz
        imp5_att, imp2_att = compute_channel_importances(feat_att, feature_names)
        imp_combined_att = {ch: (imp5_att[ch] + imp2_att[ch]) / 2 for ch in imp5_att}

        plot_channel_importance_topomap(
            imp_combined_att,
            'Transformer EEG Topomap'
        )
