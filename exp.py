import numpy as np
from sklearn import manifold
from matplotlib import pyplot as plt
plt.switch_backend('agg')


def visualizeData(Z, labels, num_clusters, title):
    '''
   TSNE visualization of the points in latent space Z
   :param Z: Numpy array containing points in latent space in which clustering was performed
   :param labels: True labels - used for coloring points
   :param num_clusters: Total number of clusters
   :param title: filename where the plot should be saved
   :return: None - (side effect) saves clustering visualization plot in specified location
    '''
    # labels = labels.astype(int)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    Z_tsne = tsne.fit_transform(Z)
    fig = plt.figure()
    plt.scatter(Z_tsne[:, 0], Z_tsne[:, 1], s=8, c=labels, cmap=plt.cm.get_cmap("jet", num_clusters))
    # plt.colorbar(ticks=range(num_clusters))  # 加柱状标签
    # plt.axis('off')  # 去掉坐标轴
    fig.savefig('A_plot_scatter/{}.pdf'.format(title), dpi=300)

    # ts = manifold.TSNE(n_components=2)
    # ts.fit_transform(Z)
    # x = ts.embedding_
    # y = labels
    # xi = []
    # for i in range(y.max()+1):
    #     xi.append(x[np.where(y == i)])
    # colors = ['mediumblue', 'green', 'red', 'yellow', 'cyan', 'mediumvioletred', 'mediumspringgreen']
    # plt.figure()
    # for i in range(7):
    #     plt.scatter(xi[i][:, 0], xi[i][:, 1], s=30, color=colors[i], marker='o', alpha=1, zorder=2)
    # plt.savefig('A_plot_scatter/{}.pdf'.format(title), dpi=300)


def plot_loss(n, xlabel, ylabel, title):
    y = []
    for i in range(1, n+1):
        enc = np.load('LossAll/epoch_{}.npy'.format(i))
        tempy = enc.tolist()
        y.append(tempy)
    x = range(1, len(y)+1)
    plt.plot(x, y, '-',  label='ours_loss')
    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.savefig('A_plot_scatter/{}.pdf'.format(title), dpi=300)


def plot_acc(n, xlabel, ylabel, title):
    y = []
    for i in range(1, n+1):
        enc = np.load('Acctrain/epoch_{}.npy'.format(i))
        tempy = enc.tolist()
        y.append(tempy)
    x = range(1, len(y)+1)
    plt.plot(x, y, '-',  label='ours_acc')
    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.savefig('A_plot_scatter/{}.pdf'.format(title), dpi=300)


def plot_embedding_2d(X, y, title):
    tsne2d = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne_2d = tsne2d.fit_transform(X)
    X = X_tsne_2d[:,0:2]
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    # Plot colors numbers
    plt.figure(figsize=(10,10))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set3(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    plt.savefig(title, dpi=300)

    '''
def visualizeData(Z, labels, num_clusters, title):

    color = []
    for i in range(Z.shape[0]):
        if labels[i] == 0:
            color.append('#FF0000')
        else:
            color.append('#000000')
    labels = labels.astype(int)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    Z_tsne = tsne.fit_transform(Z)
    fig = plt.figure()
    # plt.rcParams['figure.dpi'] = 200
    plt.scatter(Z_tsne[:, 0], Z_tsne[:, 1], s=10, c=color, cmap=plt.cm.get_cmap("jet", num_clusters))
    ax1 = plt.subplot()
    ax1.set_title('AEMVC', fontsize=10)
    # plt.gcf()   # 生成画布的大小
    # plt.grid()  # 生成网格
    # plt.colorbar(ticks=range(num_clusters))
    fig.savefig(title, dpi=200)
    '''


