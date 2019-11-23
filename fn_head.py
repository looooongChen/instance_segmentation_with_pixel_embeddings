import tensorflow as tf

def build_embedding_head(features, embedding_dim=16, name='embedding_branch'):

    with tf.variable_scope(name):
        features = tf.keras.layers.Dropout(rate=0.5)(features)
        emb = tf.keras.layers.Conv2D(embedding_dim, 3, activation='linear',padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(features)
        emb = tf.nn.l2_normalize(emb, axis=-1)

    return emb

def build_dist_head(features, name='dist_regression'):
    with tf.variable_scope(name):
        features = tf.keras.layers.Dropout(rate=0.5)(features)
        dist = tf.keras.layers.Conv2D(1, 3, activation='relu', padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(features)

    return dist

# def build_embedding_head(features, embedding_dim=16, name='embedding_branch'):

#     with tf.variable_scope(name):
#         e_conv = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(features)
#         e_conv = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(e_conv)
#         e_conv = tf.keras.layers.Dropout(rate=0.3)(e_conv)
#         emb = tf.keras.layers.Conv2D(embedding_dim, 3, activation='linear',padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(e_conv)
#         emb = tf.nn.l2_normalize(emb, axis=-1)

#     return emb

# def build_dist_head(features, name='dist_regression'):
#     with tf.variable_scope(name):
#         d_conv = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(features)
#         d_conv = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(d_conv)
#         d_conv = tf.keras.layers.Dropout(rate=0.3)(d_conv)
#         dist = tf.keras.layers.Conv2D(1, 3, activation='relu', padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(d_conv)

#     return dist
