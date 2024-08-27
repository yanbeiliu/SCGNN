from __future__ import division
from __future__ import print_function
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import accuracy, sparse_mx_to_torch_sparse_tensor, normalize_features
from models import GCN
import process
from exp import visualizeData, plot_loss, plot_acc
import numpy
from sklearn.metrics import f1_score

# np.random.seed(42)
# torch.manual_seed(42)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(42)
# ==================================================================================================================
# ==================================================================================================================
torch.cuda.set_device(0)
nb_epochs = 320
lr = 0.005
hid_units = 128
dro = 0.96
sparse = False
# sparse = True

dataset = 'citeseer'

# adj, b, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)

# features = features.todense()
# labels = np.where(labels)[1]
# visualizeData(np.array(features), labels, 7, 'Original2')
# print('done')

# diff = process.compute_ppr(b, 0.2)
# np.save('/home/user/ZSC/ketifirst2d/pygcn/data_xy/alldiff/citeseerdiff.npy', diff)
# diff = np.load('data_xy/alldiff/citeseerdiff.npy')
diff1 = np.load('data_xy/alldiff/diff.npy')


# diff = process.compute_heat(b, 5)
# np.save('/home/user/ZSC/ketifirst2d/pygcn/data_xy/alldiff/citeseerdiff_heat.npy', diff)
diff = np.load('data_xy/alldiff/citeseerdiff_heat.npy')

adj, b, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)


if sparse:
    diff = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(diff))
else:
    diff = torch.FloatTensor(diff)
    diff1 = torch.FloatTensor(diff1)

features, _ = process.preprocess_features(features)
labels = torch.FloatTensor(labels)
nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = labels.shape[1]

adj = process.normalize_adj(adj + 1 * sp.eye(adj.shape[0]))  # D^-1/2（A+In）D^-1/2
# adj = normalize_features(adj + 1 * sp.eye(adj.shape[0]))  # D^-1（A+In）

if sparse:
    adj = process.sparse_mx_to_torch_sparse_tensor(adj)
else:
    # adj = torch.FloatTensor(np.array(adj.todense()))
    adj = torch.FloatTensor(np.array((adj + sp.eye(adj.shape[0])).todense()))

features = torch.FloatTensor(features[np.newaxis])
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

batch_size = 1

lbl_1 = torch.ones(batch_size, nb_nodes * 2)
lbl_2 = torch.zeros(batch_size, nb_nodes * 2)
lbl = torch.cat((lbl_1, lbl_2), 1)

model = GCN(nfeat=ft_size,
            nhid=hid_units,
            nclass=nb_classes,
            dropout=dro)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

if torch.cuda.is_available():
    model.cuda()
    labels = labels.cuda()
    features = features.cuda()
    adj = adj.cuda()
    diff = diff.cuda()
    diff1 = diff1.cuda()
    lbl = lbl.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

b_xent = nn.BCEWithLogitsLoss()

train_lbl = torch.argmax(labels[idx_train], dim=1)
test_lbl = torch.argmax(labels[idx_test], dim=1)
if torch.cuda.is_available():
    train_lbl = train_lbl.cuda()
    test_lbl = test_lbl.cuda()


def train():
    global features
    model.train()
    optimizer.zero_grad()

    idx = np.random.permutation(nb_nodes)
    global shuf_fts
    shuf_fts = features[:, idx, :]
    shuf_fts = torch.squeeze(shuf_fts)
    features = torch.squeeze(features)

    output, logits = model(features, diff1, diff, shuf_fts, sparse, None, None, None)
    loss = F.nll_loss(output[idx_train], train_lbl) + b_xent(logits, lbl)
    # Loss = loss.cuda().data.cpu().detach().numpy()
    # np.save('LossAll/epoch_{}'.format(epoch), Loss)

    acc_train = accuracy(output[idx_train], train_lbl)
    loss.backward()
    optimizer.step()
    features = torch.unsqueeze(features, 0)
    return acc_train, loss


def test():

    model.eval()
    output, _ = model(torch.squeeze(features), diff1, diff, shuf_fts, sparse, None, None, None)
    acc_test = accuracy(output[idx_test], test_lbl)

    label_max = []
    for idx in idx_test:
        label_max.append(torch.argmax(output[idx]).item())
    labelcpu = test_lbl.data.cpu()
    macro_f1 = f1_score(labelcpu, label_max, average='macro')

    return acc_test, macro_f1, output


acc_max = 0
f1_max = 0
epoch_max = 0
for epoch in range(1, nb_epochs+1):
    train_acc, train_loss = train()
    test_acc, test_f1, _ = test()
    log = 'Epoch: {:03d}, Train_acc: {:.4f}, Train_loss: {:.4f}, Test_acc: {:.4f}, Test_f1: {:.4f}'
    print(log.format(epoch, train_acc, train_loss,  test_acc, test_f1))
    if test_acc >= acc_max:
        acc_max = test_acc
        f1_max = test_f1
        epoch_max = epoch
        torch.save(model.state_dict(), 'best_Citeseer_model.pkl')
print('Epoch: {}'.format(epoch_max),
      'acc_max: {:.4f}'.format(acc_max),
      'f1_max: {:.4f}'.format(f1_max))

model.load_state_dict(torch.load('best_Citeseer_model.pkl'))
test_accc, test_f11, output = test()
# visualizeData(output[idx_test].cuda().data.cpu().detach().numpy(),
#               test_lbl.cuda().data.cpu().detach().numpy(), 6, 'my_Citeseer{}'.format(epoch_max))
print('Test_acc: {:.4f}, Test_f1: {:.4f}'.format(test_accc, test_f11))







