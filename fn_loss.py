import tensorflow as tf

def weight_fg(label):
    """
    label: [B W H 1]
    """
    pos = tf.greater(label, 0)
    neg = tf.equal(label, 0)
    num_pos = tf.count_nonzero(pos, axis=[1,2,3], keepdims=True, dtype=tf.float32)
    num_neg = tf.count_nonzero(neg, axis=[1,2,3], keepdims=True, dtype=tf.float32)
    total = num_neg + num_pos
    return tf.cast(pos, dtype=tf.float32)*total/(2*num_pos) \
           + tf.cast(neg, dtype=tf.float32)*total/(2*num_neg)

def build_dist_loss(dist, dist_gt, name='dist_reg_loss'):

    with tf.variable_scope(name):
        weights = weight_fg(dist_gt)
        dist_gt = dist_gt * 10
        loss = tf.square(dist-dist_gt)*weights
        # loss = tf.square(dist-dist_gt)

        return tf.reduce_mean(loss)

def build_embedding_loss(embedding, label_map, neighbor, include_bg=True, name='emb_loss'):
    """
    :param embedding: [B W H C]
    :param label_map: [B W H 1]
    :param neighbor: neighbot list
    :param include_bg: weather take background as an independent object
    """

    with tf.variable_scope(name):

        def cond(loss, embedding, label_map, neighbor, i):
            return tf.less(i, tf.shape(embedding)[0])

        def body(loss, embedding, label_map, neighbor, i):
            loss_single = embedding_loss_single_example(embedding[i],
                                                        label_map[i],
                                                        neighbor[i],
                                                        include_bg)

            loss = loss.write(i, loss_single)

            return loss, embedding, label_map, neighbor, i+1

        loss = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        loss, _, _, _, _ = tf.while_loop(cond, body, [loss, embedding, label_map, neighbor, 0])

        loss = loss.stack()
        loss = tf.reduce_mean(loss)
        return loss


def embedding_loss_single_example(embedding,
                                  label_map,
                                  neighbor,
                                  include_bg=True):
    """
    build embedding loss
    :param embedding: 3 dim tensor, should be normalized
    :param label_map: 3 dim tensor with 1 channel
    :param neighbor: row N is the neighbors of object N, N starts with 1, 0 indicates the background
    :param include_bg: weather take background as an independent object
    """

    # flatten the tensors
    label_flat = tf.reshape(label_map, [-1])
    embedding_flat = tf.reshape(embedding, [-1, tf.shape(embedding)[-1]])
    embedding_flat = tf.nn.l2_normalize(embedding_flat, axis=1)
    # weight_flat = tf.reshape(weight_fg(tf.expand_dims(label_map, axis=0)), [-1, 1])

    # if not include background, mask out background pixels
    if not include_bg:
        label_mask = tf.greater(label_flat, 0)
        label_flat = tf.boolean_mask(label_flat, label_mask)
        embedding_flat = tf.boolean_mask(embedding_flat, label_mask)
        # weight_flat = tf.boolean_mask(weight_flat, label_mask)

    # grouping based on labels
    unique_labels, unique_id, counts = tf.unique_with_counts(label_flat)
    counts = tf.reshape(tf.cast(counts, tf.float32), (-1, 1))
    segmented_sum = tf.unsorted_segment_sum(embedding_flat, unique_id, tf.size(unique_labels))
    # mean embedding of each instance
    mu = tf.nn.l2_normalize(segmented_sum/counts, axis=1)
    mu_expand = tf.gather(mu, unique_id)

    ##########################
    #### inner class loss ####
    ##########################

    loss_inner = tf.losses.cosine_distance(mu_expand, embedding_flat,
                                           axis=1, 
                                        #    weights=weight_flat,
                                           reduction=tf.losses.Reduction.MEAN)

    ##########################
    #### inter class loss ####
    ##########################

    # repeat mu
    instance_num = tf.size(unique_labels)
    mu_interleave = tf.tile(mu, [instance_num, 1])
    mu_rep = tf.tile(mu, [1, instance_num])
    mu_rep = tf.reshape(mu_rep, (instance_num*instance_num, -1))

    # get inter loss for each pair
    loss_inter = tf.losses.cosine_distance(mu_interleave, mu_rep,
                                           axis=1,
                                           reduction=tf.losses.Reduction.NONE)
    loss_inter = tf.abs(1-loss_inter)
    
    # compute adjacent indicator
    # indicator: bg(0) is adjacent to any object
    # 0 1 1 1 1 ...
    # 1 x x x x ...
    # 1 x x x x ...
    # ...
    bg = tf.zeros([tf.shape(neighbor)[0], 1], dtype=tf.int32)
    neighbor = tf.concat([bg, neighbor], axis=1)
    dep = instance_num if include_bg else instance_num + 1

    adj_indicator = tf.one_hot(neighbor, depth=dep, dtype=tf.float32)
    adj_indicator = tf.reduce_sum(adj_indicator, axis=1)
    adj_indicator = tf.cast(adj_indicator > 0, tf.float32)

    bg_indicator = tf.one_hot(0, depth=dep, on_value=0.0, off_value=1.0, dtype=tf.float32)
    bg_indicator = tf.reshape(bg_indicator, [1, -1])
    indicator = tf.concat([bg_indicator, adj_indicator], axis=0)

    # reorder the rows and columns in the same order of unique_labels
    # if background (0) is not included, the first row and column will be ignores, since 0 is not the unique_labels
    indicator = tf.gather(indicator, unique_labels, axis=0)
    indicator = tf.gather(indicator, unique_labels, axis=1)
    inter_mask = tf.reshape(indicator, [-1, 1])

    loss_inter = tf.reduce_sum(loss_inter*inter_mask)/(tf.reduce_sum(inter_mask)+1e-12)

    return loss_inner+loss_inter
