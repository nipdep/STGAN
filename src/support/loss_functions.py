#%%
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
import numpy as np
import math 
import random
from sklearn.model_selection import KFold
from scipy import interpolate
import config
#%%
def pairWiseRankingLoss(y_ref, y_style, label):
    m  = tf.cast(tf.broadcast_to(config.LOSS_THD, shape=[y_ref.shape[0], ]), dtype=tf.float32)
    u  = tf.cast(tf.broadcast_to(0, shape=[y_ref.shape[0], ]), dtype=tf.float32)
    i  = tf.cast(tf.broadcast_to(1, shape=[y_ref.shape[0], ]), dtype=tf.float32)
    w = tf.cast(tf.broadcast_to(2, shape=[y_ref.shape[0], ]), dtype=tf.float32)
    y = tf.cast(label, dtype=tf.float32)
    dist = tf.divide(tf.abs(tf.keras.losses.cosine_similarity(y_ref,y_style)+i), w)
    loss = tf.math.multiply(y,dist) + tf.math.multiply((i-y),-1*tf.reduce_min(tf.stack([u,m-dist]), axis=0))
    return tf.cast(tf.reduce_mean(loss), dtype=tf.float32)
    

class MarginalAcc(tf.keras.metrics.Metric):

    def __init__(self, name='marginal accuracy', **kwargs):
        super(MarginalAcc, self).__init__(name=name, **kwargs)
        self.acc = self.add_weight(name='marginal acc', initializer='zero', dtype=tf.float32)
        self.tpr = self.add_weight(name='true possitive rate', initializer='zero', dtype=tf.float32)
        self.fpr = self.add_weight(name='false possitive rate', initializer='zero', dtype=tf.float32)

    def update_state(self, y_ref, y_style, label):
        dist = distance(y_ref, y_style, 0)
        tpr, fpr, acc = calculate_accuracy(config.LOSS_THD,dist, label)
        self.acc.assign(acc)
        self.tpr.assign(tpr)
        self.fpr.assign(fpr)

    def result(self):
        return self.acc, #self.tpr, self.fpr


#%%
def SM_SSIMLoss(ref_img, gen_img):
    one = tf.cast(tf.broadcast_to(1, shape=ref_img.shape), dtype=tf.float32)
    two = tf.cast(tf.broadcast_to(2, shape=ref_img.shape), dtype=tf.float32)
    rescaled_ref_img = tf.abs(tf.divide(tf.add(one, ref_img), two))
    rescaled_gen_img = tf.abs(tf.divide(tf.add(one, gen_img), two))
    loss = tf.image.ssim_multiscale(ref_img, gen_img, max_val=2, filter_size=3)
    return tf.reduce_mean(loss)

def pairwise_distance(feature, squared=False):
    """Computes the pairwise distance matrix with numerical stability.

    output[i, j] = || feature[i, :] - feature[j, :] ||_2

    Args:
      feature: 2-D Tensor of size [number of data, feature dimension].
      squared: Boolean, whether or not to square the pairwise distances.

    Returns:
      pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """
    pairwise_distances_squared = math_ops.add(
        math_ops.reduce_sum(math_ops.square(feature), axis=[1], keepdims=True),
        math_ops.reduce_sum(
            math_ops.square(array_ops.transpose(feature)),
            axis=[0],
            keepdims=True)) - 2.0 * math_ops.matmul(feature,
                                                    array_ops.transpose(feature))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = math_ops.maximum(pairwise_distances_squared, 0.0)
    # Get the mask where the zero distances are at.
    error_mask = math_ops.less_equal(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = math_ops.sqrt(
            pairwise_distances_squared + math_ops.to_float(error_mask) * 1e-16)

    # Undo conditionally adding 1e-16.
    pairwise_distances = math_ops.multiply(
        pairwise_distances, math_ops.to_float(math_ops.logical_not(error_mask)))

    num_data = array_ops.shape(feature)[0]
    # Explicitly set diagonals to zero.
    mask_offdiagonals = array_ops.ones_like(pairwise_distances) - array_ops.diag(
        array_ops.ones([num_data]))
    pairwise_distances = math_ops.multiply(pairwise_distances, mask_offdiagonals)
    return pairwise_distances

def masked_maximum(data, mask, dim=1):
    """Computes the axis wise maximum over chosen elements.

    Args:
      data: 2-D float `Tensor` of size [n, m].
      mask: 2-D Boolean `Tensor` of size [n, m].
      dim: The dimension over which to compute the maximum.

    Returns:
      masked_maximums: N-D `Tensor`.
        The maximized dimension is of size 1 after the operation.
    """
    axis_minimums = math_ops.reduce_min(data, dim, keepdims=True)
    masked_maximums = math_ops.reduce_max(
        math_ops.multiply(data - axis_minimums, mask), dim,
        keepdims=True) + axis_minimums
    return masked_maximums

def masked_minimum(data, mask, dim=1):
    """Computes the axis wise minimum over chosen elements.

    Args:
      data: 2-D float `Tensor` of size [n, m].
      mask: 2-D Boolean `Tensor` of size [n, m].
      dim: The dimension over which to compute the minimum.

    Returns:
      masked_minimums: N-D `Tensor`.
        The minimized dimension is of size 1 after the operation.
    """
    axis_maximums = math_ops.reduce_max(data, dim, keepdims=True)
    masked_minimums = math_ops.reduce_min(
        math_ops.multiply(data - axis_maximums, mask), dim,
        keepdims=True) + axis_maximums
    return masked_minimums

def triplet_loss(anchor, positive, negative, alpha):
    """Calculate the triplet loss according to the FaceNet paper
    
    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.
  
    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    with tf.name_scope('triplet_loss') as scope:
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
        
        basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
      
    return loss

def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric==0:
        # Euclidian distance
        diff = tf.math.subtract(embeddings1, embeddings2)
        dist = tf.reduce_sum(tf.math.pow(diff, 2),1)
    elif distance_metric==1:
        # Distance based on cosine similarity
        dot = tf.reduce_sum(tf.math.multiply(embeddings1, embeddings2), axis=1)
        norm = tf.norm(embeddings1, axis=1) * tf.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = tf.math.acos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric 
        
    return tf.cast(dist, dtype=tf.float32)

def pairWiseRankingLoss(y_ref, y_style, label, d_type=0):
    m  = tf.cast(tf.broadcast_to(config.LOSS_THD, shape=[y_ref.shape[0], ]), dtype=tf.float32)
    u  = tf.cast(tf.broadcast_to(0, shape=[y_ref.shape[0], ]), dtype=tf.float32)
    i  = tf.cast(tf.broadcast_to(1, shape=[y_ref.shape[0], ]), dtype=tf.float32)
    w = tf.cast(tf.broadcast_to(2, shape=[y_ref.shape[0], ]), dtype=tf.float32)
    y = tf.cast(label, dtype=tf.float32)
    dist = distance(y_ref, y_style, d_type)
    loss = tf.math.multiply(y,dist) + tf.math.multiply((i-y),tf.reduce_max(tf.stack([u,m-dist]), axis=0))
    return tf.cast(tf.reduce_mean(loss), dtype=tf.float32)

def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)
    
    tprs = np.zeros((nrof_folds,nrof_thresholds))
    fprs = np.zeros((nrof_folds,nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    
    indices = np.arange(nrof_pairs)
    
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
          mean = 0.0
        dist = distance(embeddings1-mean, embeddings2-mean, distance_metric)
        
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _ = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])
          
        tpr = np.mean(tprs,0)
        fpr = np.mean(fprs,0)
    return tpr, fpr, accuracy

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = tf.math.less(dist, threshold)
    actual_issame = tf.cast(actual_issame, dtype=tf.bool)
    tp = tf.reduce_sum(tf.cast(tf.math.logical_and(predict_issame, actual_issame), dtype=tf.int32))
    fp = tf.reduce_sum(tf.cast(tf.math.logical_and(predict_issame, tf.math.logical_not(actual_issame)), dtype=tf.int32))
    tn = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(predict_issame), tf.math.logical_not(actual_issame)), dtype=tf.int32))
    fn = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(predict_issame), actual_issame), dtype=tf.int32))
    zero = tf.constant([0], dtype=tf.float32)
    tpr = zero if tf.math.equal(tf.cast(tp+fn, dtype=tf.float32), zero) else tf.cast(tp, dtype=tf.float32) / tf.cast(tp+fn, dtype=tf.float32)
    fpr = zero if tf.math.equal(tf.cast(fp+tn, dtype=tf.float32), zero) else tf.cast(fp, dtype=tf.float32) / tf.cast(fp+tn, dtype=tf.float32)
    acc = tf.cast(tp+tn, dtype=tf.float32)/dist.shape[0]
    return tf.cast(tpr, dtype=tf.float32), tf.cast(fpr, dtype=tf.float32), tf.cast(acc, dtype=tf.float32)


  
def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10, distance_metric=0, subtract_mean=False):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)
    
    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)
    
    indices = np.arange(nrof_pairs)
    
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
          mean = 0.0
        dist = distance(embeddings1-mean, embeddings2-mean, distance_metric)
      
        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train)>=far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0
    
        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])
  
    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far