from __future__ import division
from __future__ import print_function
import torch.nn as nn
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_data, accuracy, sparse_mx_to_torch_sparse_tensor, compute_ppr, compute_heat
from models import GCN, LogReg
from exp import visualizeData, plot_loss, plot_acc, plot_embedding_2d
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, accuracy_score

# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# np.random.seed(42)
# torch.manual_seed(42)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(42)
# ==================================================================================================================
# ==================================================================================================================
torch.cuda.set_device(-1)
nb_epochs = 250
lr = 0.001
hid_units = 512
dro = 0.4
sparse = False

x = 1
# adj, b, features, labels, idx_train, idx_val, idx_test = load_data()

# ff = torch.squeeze(features)
# visualizeData(ff[idx_test.numpy()], labels[idx_test].numpy(), 7, 'Original_test')
# print('done')

# diff = compute_ppr(b, 0.2)
# np.save('/home/user/ZSC/ketifirst2d/pygcn/coradiff/diff.npy', diff)
diff1 = np.load('coradiff/diff.npy')


# diff = compute_heat(b, 5)
# np.save('/home/user/ZSC/ketifirst2d/pygcn/coradiff/diff_heat.npy', diff)
diff = np.load('coradiff/diff_heat.npy')

adj, b, features, labels, idx_train, idx_val, idx_test = load_data()


if sparse:
    diff = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(diff))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
else:
    # diff = torch.FloatTensor(diff)
    diff1 = torch.FloatTensor(diff1)
    adj = torch.FloatTensor(np.array(adj.todense()))

ft_size = features.shape[2]  # 1433
nb_classes = np.unique(labels).shape[0]  # 7
nb_nodes = features.shape[1]   # 2708
batch_size = 1

lbl_1 = torch.ones(batch_size, nb_nodes * 2)
lbl_2 = torch.zeros(batch_size, nb_nodes * 2)
lbl = torch.cat((lbl_1, lbl_2), 1)

model = GCN(nfeat=ft_size,
            nhid=hid_units,
            nclass=nb_classes,
            dropout=dro)

# if torch.cuda.device_count() > 1:
#     print("Use", torch.cuda.device_count(), 'gpus')
#     model = nn.DataParallel(model, device_ids=[0, 1])
# model.to(device)
# features = features.to(device)
# adj = adj.to(device)
# labels = labels.to(device)
# idx_val = idx_val.to(device)
# idx_test = idx_test.to(device)
# idx_train = idx_train.to(device)
# lbl = lbl.to(device)
# diff = diff.to(device)

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


def train():
    global features
    model.train()
    optimizer.zero_grad()

    idx = np.random.permutation(nb_nodes)
    global shuf_fts
    shuf_fts = features[:, idx, :]
    shuf_fts = torch.squeeze(shuf_fts)

    features = torch.squeeze(features)

    output, logits = model(features, adj, diff1, shuf_fts, sparse, None, None, None)
    loss = F.nll_loss(output[idx_train], labels[idx_train]) + b_xent(logits, lbl)
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss.backward()
    optimizer.step()

    features = torch.unsqueeze(features, 0)
    # visualizeData(output[idx_train].data.cpu().numpy(),
    #               labels[idx_train].data.cpu().numpy(), 7, 'idx_train_Result{}'.format(epoch))

    # visualizeData(output.cuda().data.cpu().detach().numpy(),
    #               labels.cuda().data.cpu().detach().numpy(), 7, 'Result')

    return acc_train, loss


def test():
    model.eval()
    output, _ = model(torch.squeeze(features), adj, diff1, shuf_fts, sparse, None, None, None)
    acc_test = accuracy(output[idx_test], labels[idx_test])

    # x = np.arange(1,1001)
    # preds = output[idx_test].max(1)[1].type_as(labels)
    # plt.figure()
    # plt.plot(x, labels[idx_test].cuda().data.cpu().detach().numpy(), ls="-", marker=" ", color="r")
    # plt.plot(x, preds.cuda().data.cpu().detach().numpy(), ls="-", marker=" ", color="b")
    # plt.legend(["target", "prediction"], loc="upper right")
    # plt.savefig('m.png')

    label_max = []
    for idx in idx_test:
        label_max.append(torch.argmax(output[idx]).item())
    labelcpu = labels[idx_test].data.cpu()
    macro_f1 = f1_score(labelcpu, label_max, average='macro')

    # plot_embedding_2d(output[idx_test].cuda().data.cpu().detach().numpy(),
    #                   labels[idx_test].cuda().data.cpu().detach().numpy(),'my_cora{}'.format(epoch))
    return acc_test, macro_f1, output


acc_max = 0
f1_max = 0
epoch_max = 0

# aa=[]

for epoch in range(1, nb_epochs+1):
    train_acc, train_loss = train()

    # a=train_loss.data.cpu().detach().numpy()
    # a=a.tolist()
    # aa.append(a)

    test_acc, test_f1, _ = test()
    log = 'Epoch: {:03d}, Train_acc: {:.4f}, Train_loss: {:.4f}, Test_acc: {:.4f}, Test_f1: {:.4f}'
    print(log.format(epoch, train_acc, train_loss,  test_acc, test_f1))
    if test_acc >= acc_max:
        acc_max = test_acc
        f1_max = test_f1
        epoch_max = epoch
        torch.save(model.state_dict(), 'best_Cora_model.pkl')


# aa=np.array(aa)
# np.savetxt('aa.csv',aa)

print('Epoch: {}'.format(epoch_max),
      'acc_max: {:.4f}'.format(acc_max),
      'f1_max: {:.4f}'.format(f1_max))

model.load_state_dict(torch.load('best_Cora_model.pkl'))
test_accc, test_f11, output = test()
# visualizeData(output[idx_test].cuda().data.cpu().detach().numpy(),
#                   labels[idx_test].cuda().data.cpu().detach().numpy(), 7, 'my_cora{}'.format(epoch_max))
print('Test_acc: {:.4f}, Test_f1: {:.4f}'.format(test_accc, test_f11))


