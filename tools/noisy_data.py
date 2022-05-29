import torch
import torch.nn.functional as F

import numpy as np
from math import inf
from scipy import stats
from numpy.testing import assert_array_almost_equal


def dataset_split(train_images, train_labels, noise_rate=0.5, noise_type='symmetric', split_per=0.9, seed=1, num_classes=10):
    '''
    :param train_images: array
    :param train_labels: array
    :param noise_rate:
    :param noise_type: only support symmetric noise, pairflip (asymmetric), instance
    :param split_per: proportion for data used for training, rest are for validation
    :param seed: random seed for spliting train and validation; also random seed for creating random noises
    :param num_classes:
    :return: all are arrays
    '''
    assert noise_type in ['symmetric', 'pairflip', 'instance'], "not supported noise type"
    clean_train_labels = train_labels[:, np.newaxis] # 1 dim -> 2 dim
    if(noise_type == 'pairflip'):
        noisy_labels, real_noise_rate, transition_matrix = noisify_pairflip(clean_train_labels, noise=noise_rate, seed=seed, nb_classes=num_classes)
    elif(noise_type == 'symmetric'):
        noisy_labels, real_noise_rate, transition_matrix = noisify_symmetric(clean_train_labels, noise=noise_rate, seed=seed, nb_classes=num_classes)
    else: # instance
        norm_std = 0.1
        if(len(train_images.shape) == 2):
            feature_size = train_images.shape[1]
        else:
            feature_size = 1
            for i in range(1, len(train_images.shape)):
                feature_size = int(feature_size * train_images.shape[i])

        if torch.is_tensor(train_images) is False:
            data = torch.from_numpy(train_images)
        else:
            data = train_images

        data = data.type(torch.FloatTensor)
        targets = torch.from_numpy(train_labels) # 把data和target全都变成tensor放进函数get_instance_noisy_label 为了矩阵计算快一些
        dataset = zip(data, targets)
        noisy_labels = get_instance_noisy_label(noise_rate, dataset, targets, num_classes, feature_size, norm_std, seed)

    clean_train_labels = clean_train_labels.squeeze()
    noisy_labels = noisy_labels.squeeze() # 2 dim -> 1 dim
    num_samples = int(noisy_labels.shape[0])
    np.random.seed(seed)
    train_set_index = np.random.choice(num_samples, int(num_samples * split_per), replace=False)
    index = np.arange(train_images.shape[0])
    val_set_index = np.delete(index, train_set_index)

    train_set, val_set = train_images[train_set_index, :], train_images[val_set_index, :]
    train_labels, val_labels = noisy_labels[train_set_index], noisy_labels[val_set_index]
    train_clean_labels, val_clean_labels = clean_train_labels[train_set_index], clean_train_labels[val_set_index]
    # train_absolute_paths, val_absolute_paths = train_paths[train_set_index], train_paths[val_set_index]
    return train_set, val_set, train_labels, val_labels, train_clean_labels, val_clean_labels


def noisify_pairflip(y_train, noise, seed=1, nb_classes=10):
    """flip label 0->1, 1->2, ... , n-1->n, n->0
    """
    P = np.eye(nb_classes)
    n = noise
    actual_noise = 0
    y_train_noisy = y_train

    if n > 0.0:
        for i in range(nb_classes - 1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes - 1, nb_classes - 1], P[nb_classes - 1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P, seed=seed)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Noisy label transition matrix: \n', P)

    return y_train_noisy, actual_noise, P


def noisify_symmetric(y_train, noise, seed=1, nb_classes=10):
    """symmetric noise
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P
    actual_noise = 0
    y_train_noisy = y_train

    if n > 0.0:
        for i in range(nb_classes):
            P[i, i] = 1. - n

        y_train_noisy = multiclass_noisify(y_train, P=P, seed=seed)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Noisy label transition matrix: \n', P)

    return y_train_noisy, actual_noise, P


def get_instance_noisy_label(noise_rate, dataset, labels, num_classes, feature_size, norm_std, seed):
    '''instance dependent noise
    inputs dataset, labels are tensors
    '''
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed(int(seed))

    P = [] # transition matrix
    flip_distribution = stats.truncnorm((0 - noise_rate) / norm_std, (1 - noise_rate) / norm_std, loc=noise_rate, scale=norm_std) # cut [0, 1] for N(noise, norm_std)
    flip_rate = flip_distribution.rvs(labels.shape[0]) # produce samples following truncated distribution

    if torch.cuda.is_available():
        labels = labels.cuda()  # 放入cuda是想着矩阵计算快一些吧
    W = np.random.randn(num_classes, feature_size, num_classes) # 3 dim array
    if torch.cuda.is_available():
        W = torch.FloatTensor(W).cuda()
    else:
        W = torch.FloatTensor(W)
    for i, (x, y) in enumerate(dataset):
        if torch.cuda.is_available():
            x = x.cuda()
            x = x.reshape(feature_size)
        else:
            x = x.reshape(feature_size) # 把图片给拉直成一个1 dim的向量

        A = x.view(1, -1).mm(W[y]).squeeze(0) # 两个矩阵相乘 (1, feature_size) X (feature_size, num_classes)
        A[y] = -inf #
        A = flip_rate[i] * F.softmax(A, dim=0)
        A[y] += 1 - flip_rate[i]
        # 相当于是 对于每个sample的label 以flip_rate[i](random sample from truncated normal distribution)的概率翻转到另外的label
        # 翻转到其他的label的概率又和sample的所有feature是有关系的（矩阵相乘）
        P.append(A)
    P = torch.stack(P, 0).cpu().numpy()
    l1 = [i for i in range(num_classes)]
    new_y = [np.random.choice(l1, p=P[i]) for i in range(labels.shape[0])]
    # print('Noisy label transition matrix: ', P) 这里的P和之前的P都不一样了  这里的P的size是 #samples*num_classes

    record = [[0 for _ in range(num_classes)] for i in range(num_classes)]
    for a, b in zip(labels, new_y):
        a, b = int(a), int(b)
        record[a][b] += 1

    # print('Label transition Conclusion: ', record)
    print('instance-dependent noisy data created')
    return np.array(new_y)


def multiclass_noisify(y, P, seed=1):
    '''
    :param y: true labels
    :param P: transition matrix
    :param random_state: seed
    :return: noisy labels
    '''
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy() # noisy labels
    flipper = np.random.RandomState(seed)

    for idx in np.arange(m):
        i = y[idx]
        # i is np.array, such as [1]
        if not isinstance(i, np.ndarray):
            i = [i]
        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y
