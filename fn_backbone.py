import tensorflow as tf

DROP_RATE=0.5

def build_doubleHead(inputs, features=32, drop_rate=0.2, name="doubleHead_UNet"):

    with tf.variable_scope(name):
        with tf.variable_scope("Conv1"):
            conv1 = tf.keras.layers.Conv2D(features, 3, activation='relu', padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(inputs)
            conv1 = tf.keras.layers.Conv2D(features, 3, activation='relu', padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(conv1)
            pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        with tf.variable_scope("Conv2"):
            conv2 = tf.keras.layers.Conv2D(2*features, 3, activation='relu', padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(pool1)
            conv2 = tf.keras.layers.Conv2D(2*features, 3, activation='relu', padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(conv2)
            pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
        with tf.variable_scope("Conv3"):
            conv3 = tf.keras.layers.Conv2D(4*features, 3, activation='relu', padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(pool2)
            conv3 = tf.keras.layers.Conv2D(4*features, 3, activation='relu', padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(conv3)
            pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
        with tf.variable_scope("Conv4"):
            conv4 = tf.keras.layers.Conv2D(8*features, 3, activation='relu', padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(pool3)
            conv4 = tf.keras.layers.Conv2D(8*features, 3, activation='relu', padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(conv4)
            drop4 = tf.keras.layers.Dropout(drop_rate)(conv4)
            pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)
        with tf.variable_scope("Conv5"):
            conv5 = tf.keras.layers.Conv2D(16*features, 3, activation='relu', padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(pool4)
            conv5 = tf.keras.layers.Conv2D(16*features, 3, activation='relu', padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(conv5)
            drop5 = tf.keras.layers.Dropout(drop_rate)(conv5)

        with tf.variable_scope("Conv6_1"):
            up6_1 = tf.keras.layers.Conv2D(8*features, 2, activation='relu',padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(
                tf.keras.layers.UpSampling2D(size=(2, 2))(drop5))
            merge6_1 = tf.keras.layers.concatenate([drop4, up6_1], axis=3)
            conv6_1 = tf.keras.layers.Conv2D(8*features, 3, activation='relu', padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(merge6_1)
            conv6_1 = tf.keras.layers.Conv2D(8*features, 3, activation='relu', padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(conv6_1)
        with tf.variable_scope("Conv6_2"):
            up6_2 = tf.keras.layers.Conv2D(8*features, 2, activation='relu',padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(
                tf.keras.layers.UpSampling2D(size=(2, 2))(drop5))
            merge6_2 = tf.keras.layers.concatenate([drop4, up6_2], axis=3)
            conv6_2 = tf.keras.layers.Conv2D(8*features, 3, activation='relu', padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(merge6_2)
            conv6_2 = tf.keras.layers.Conv2D(8*features, 3, activation='relu', padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(conv6_2)


        with tf.variable_scope("Conv7_1"):
            up7_1 = tf.keras.layers.Conv2D(4*features, 2, activation='relu', padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(
                tf.keras.layers.UpSampling2D(size=(2, 2))(conv6_1))
            merge7_1 = tf.keras.layers.concatenate([conv3, up7_1], axis=3)
            conv7_1 = tf.keras.layers.Conv2D(4*features, 3, activation='relu', padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(merge7_1)
            conv7_1 = tf.keras.layers.Conv2D(4*features, 3, activation='relu', padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(conv7_1)
        with tf.variable_scope("Conv7_2"):
            up7_2 = tf.keras.layers.Conv2D(4*features, 2, activation='relu', padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(
                tf.keras.layers.UpSampling2D(size=(2, 2))(conv6_2))
            merge7_2 = tf.keras.layers.concatenate([conv3, up7_2], axis=3)
            conv7_2 = tf.keras.layers.Conv2D(4*features, 3, activation='relu', padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(merge7_2)
            conv7_2 = tf.keras.layers.Conv2D(4*features, 3, activation='relu', padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(conv7_2)

        with tf.variable_scope("Conv8_1"):
            up8_1 = tf.keras.layers.Conv2D(2*features, 2, activation='relu', padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(
                tf.keras.layers.UpSampling2D(size=(2, 2))(conv7_1))
            merge8_1 = tf.keras.layers.concatenate([conv2, up8_1], axis=3)
            conv8_1 = tf.keras.layers.Conv2D(2*features, 3, activation='relu', padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(merge8_1)
            conv8_1 = tf.keras.layers.Conv2D(2*features, 3, activation='relu', padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(conv8_1)
        with tf.variable_scope("Conv8_2"):
            up8_2 = tf.keras.layers.Conv2D(2*features, 2, activation='relu', padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(
                tf.keras.layers.UpSampling2D(size=(2, 2))(conv7_2))
            merge8_2 = tf.keras.layers.concatenate([conv2, up8_2], axis=3)
            conv8_2 = tf.keras.layers.Conv2D(2*features, 3, activation='relu', padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(merge8_2)
            conv8_2 = tf.keras.layers.Conv2D(2*features, 3, activation='relu', padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(conv8_2)
        
        with tf.variable_scope("Conv9_1"):
            up9_1 = tf.keras.layers.Conv2D(features, 2, activation='relu', padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(
                tf.keras.layers.UpSampling2D(size=(2, 2))(conv8_1))
            merge9_1 = tf.keras.layers.concatenate([conv1, up9_1], axis=3)
            conv9_1 = tf.keras.layers.Conv2D(features, 3, activation='relu', padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(merge9_1)
            conv9_1 = tf.keras.layers.Conv2D(features, 3, activation='relu', padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(conv9_1)
        with tf.variable_scope("Conv9_2"):
            up9_2 = tf.keras.layers.Conv2D(features, 2, activation='relu', padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(
                tf.keras.layers.UpSampling2D(size=(2, 2))(conv8_2))
            merge9_2 = tf.keras.layers.concatenate([conv1, up9_2], axis=3)
            conv9_2 = tf.keras.layers.Conv2D(features, 3, activation='relu', padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(merge9_2)
            conv9_2 = tf.keras.layers.Conv2D(features, 3, activation='relu', padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(conv9_2)

    return conv9_1, conv9_2