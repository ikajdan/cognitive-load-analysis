from imports import *
from additional import *
from cnn import MultiHeadConv1DModel  
from mlp import MLPModel
from transformer import FeatureGroupTransformerModel
from utils.data_load import load_all_participants

all_participants_data = load_all_participants()

def plot_feature_importances(names, importances, top_n=20, title='Feature Importances'):
    items = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)[:top_n]
    feats, vals = zip(*items)
    plt.figure(figsize=(10, 8))
    plt.barh(feats[::-1], vals[::-1])
    plt.title(title)
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()

def compute_model_importances(
    all_data,
    model_type='transformer',    # 'mlp', 'cnn', 'transformer'
    task_type='binary',          # 'binary', 'ternary', 'continuous'
    batch_size=32,
    learning_rate=1e-4,
    quick_train_epochs=1,
    shap_nsamples=200,
    subject_calibration=0.0
):
    # 1) Data prep
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
        # get background & test tensors
        Xbg = next(iter(bg_dl))[0].cpu().float()
        Xts = next(iter(te_dl))[0].cpu().float()

        # use DeepExplainer on the MLP
        explainer = shap.DeepExplainer(m_cpu, Xbg)
        shap_vals = explainer.shap_values(Xts)

        # stack per-class arrays or take the single array
        if isinstance(shap_vals, (list, tuple)):
            arr = np.vstack([
                v if isinstance(v, np.ndarray) else v.cpu().numpy()
                for v in shap_vals
            ])
        else:
            arr = (
                shap_vals
                if isinstance(shap_vals, np.ndarray)
                else shap_vals.cpu().numpy()
            )

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
        grp = attn.mean(dim=0).numpy()  # (n_groups,)
        imp = np.zeros(len(feature_names))
        for i,(nm,(st,sz)) in enumerate(feature_groups.items()):
            imp[st:st+sz] = grp[i]/sz
        return attn.cpu().numpy(), imp

    # select
    if model_type=='mlp':
        imp_fn = mlp_sampling
    elif model_type=='cnn':
        imp_fn = cnn_grad
    else:
        imp_fn = transformer_attn

    all_feats, all_groups = [], []
    extra_info = None

    for train_idx, test_idx in tqdm(gkf.split(X_flat,y,groups), desc="CV folds", total=len(np.unique(groups))):
        # calibration
        if subject_calibration>0:
            ncal = int(len(test_idx)*subject_calibration)
            if ncal>0:
                cal = np.random.choice(test_idx,ncal,replace=False)
                train_idx = np.concatenate([train_idx,cal])
                test_idx = np.setdiff1d(test_idx,cal)
        Xtr,Xte = X_flat[train_idx], X_flat[test_idx]
        ytr,yte = y[train_idx], y[test_idx]
        scaler=StandardScaler().fit(Xtr)
        Xtr,Xte=scaler.transform(Xtr),scaler.transform(Xte)
        tr_dl=DataLoader(create_torch_dataset(Xtr,ytr,task_type),batch_size=batch_size,shuffle=True)
        te_dl=DataLoader(create_torch_dataset(Xte,yte,task_type),batch_size=batch_size,shuffle=False)

        # model instantiation
        out_dim = 3 if task_type=='ternary' else 1
        if model_type=='mlp':
            model=MLPModel(input_size=Xtr.shape[1],output_size=out_dim,task_type=task_type)
        elif model_type=='cnn':
            model=MultiHeadConv1DModel(feature_groups,output_size=out_dim,task_type=task_type)
        else:
            model=FeatureGroupTransformerModel(feature_groups,output_size=out_dim,task_type=task_type)
        model.to(device)
        opt=torch.optim.Adam(model.parameters(),lr=learning_rate)

        # quick train
        model.train()
        for _ in range(quick_train_epochs):
            for xb,yb in tr_dl:
                xb,yb=xb.to(device),yb.to(device)
                opt.zero_grad()
                loss=loss_fn(model(xb),yb)
                loss.backward(); opt.step()

        # compute importances + extra
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
        all_groups.append(feat_imp)  # groups aggregated later

    feat_imp = np.mean(np.stack(all_feats),axis=0)
    group_imp={nm:feat_imp[st:st+sz].mean() for nm,(st,sz) in feature_groups.items()}

    return feat_imp, group_imp, extra_info, feature_names


model_type='mlp'
feat_imp, group_imp, info ,feature_names= compute_model_importances(
    all_data=all_participants_data,
    model_type=model_type,
    task_type='binary',
    batch_size=64,
    learning_rate=1e-4,
    quick_train_epochs=1,
    shap_nsamples=100,
    subject_calibration=0.00
)

feature_groups = create_token_groups(all_participants_data)
names, vals = zip(*sorted(group_imp.items(), key=lambda x: -x[1]))
plt.figure(figsize=(12,9))
plt.barh(names[::-1], vals[::-1])
plt.title("Top Groups")
plt.tight_layout()
plt.show()

# plot feature importances
def visualize_feature_importances(info, model_type,all_participants_data):
    feature_groups = create_token_groups(all_participants_data)
    names, vals = zip(*sorted(group_imp.items(), key=lambda x: -x[1]))
    plt.figure(figsize=(12,9))
    plt.barh(names[::-1], vals[::-1])
    plt.title("Top Groups")
    plt.tight_layout()
    plt.show()
    if isinstance(info, tuple) and len(info) == 2 and model_type == 'mlp':
        shap_vals, Xts = info
        # aggregate SHAP values per feature group
        group_shap = np.stack([
            shap_vals[:, st:st+sz].mean(axis=1)
            for _, (st, sz) in feature_groups.items()
        ], axis=1)  # shape: (n_samples, n_groups)
        # violin plot per group
        shap.summary_plot(
            group_shap,
            group_shap,
            feature_names=list(feature_groups.keys()),
            plot_type="violin"
        )

    elif model_type == 'cnn':
        grads = info
        plt.figure(figsize=(24, 4))
        ax = sns.heatmap(grads[0:1], cbar=True, cmap='coolwarm')
        ticks, labels = [], []
        for name, (st, sz) in feature_groups.items():
            ticks.append(st + sz // 2)
            labels.append(name)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticks([0])
        ax.set_yticklabels(['grad'])
        plt.title("CNN per-group gradient magnitude")
        plt.tight_layout()
        plt.show()
    else:
        attn = info
        corr = np.corrcoef(attn, rowvar=False)
        labels = list(feature_groups.keys())
        plt.figure(figsize=(16, 12))
        sns.heatmap(corr, xticklabels=labels, yticklabels=labels, cmap='coolwarm', center=0)
        plt.title("Attention-weight Correlation Across Groups")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

visualize_feature_importances(info, model_type, all_participants_data)
