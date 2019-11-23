import os
import tensorflow as tf
import numpy as np
from Net import LocalDisNet
from fn_backbone import build_unet, build_unet_d7
# from data_utils import extract_fn
# import cv2
# from utils.evaluation import Evaluator


def main(_):
    tf_flags = tf.app.flags.FLAGS
    # gpu config.
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    # config.gpu_options.allow_growth = True

    if tf_flags.architecture == 'd7':
        backbone_net = build_unet_d7
    else:
        backbone_net = build_unet

    if tf_flags.phase == "train":
        with tf.Session() as sess:
            train_model = LocalDisNet(sess, backbone_net, tf_flags)
            train_model.train(tf_flags.batch_size,
                              tf_flags.training_epoches,
                              os.path.join(tf_flags.train_dir),
                              os.path.join(tf_flags.val_dir))
    elif tf_flags.phase == 'test':
        # import skimage as ski
        from skimage import morphology, measure
        from skimage.io import imsave

        if not os.path.exists(tf_flags.res_dir):
            os.makedirs(tf_flags.res_dir)

        e = Evaluator()

        # load dataset from tfrecords
        val_dir = os.path.join(tf_flags.dataset_dir, 'test')
        val_tf = [os.path.join(val_dir, f) for f in os.listdir(val_dir)]
        # build dataset
        val_ds = tf.data.TFRecordDataset(val_tf)
        val_ds = val_ds.map(lambda sample:
                            extract_fn(sample, [512, 512], augmentation=False, return_raw=True)).batch(1)
        val_iterator = val_ds.make_one_shot_iterator()
        val_example = val_iterator.get_next()

        with tf.Session() as sess:
            # test on a image pair.
            import csv
            csvfile = open('./res_cells.csv', 'w', newline='')
            csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

            model = net.DiscrimitiveNet(sess, backbone_net, tf_flags)

            while True:
                try:
                    img, gt, dist, _, example_raw = sess.run(val_example)
                    _, fname = os.path.split(example_raw['image/filename'][0].decode('utf-8'))

                    # h, w = example_raw['image/height'][0], example_raw['image/width'][0]
                    pred, emb = model._segment_emb(img, dist)

                    aps = e.add_example(np.squeeze(pred), np.squeeze(gt))
                    print(['%.2f' % ap for ap in aps])

                    aps_b = ['%.4f' % ap for ap in aps]
                    csvwriter.writerow([fname] + aps_b)
                    e.save_last_as_image(os.path.join(tf_flags.res_dir, fname),
                                         img[0, :, : , :], thres=0.6, isBGR=False)

                    # center = np.squeeze(center)
                    # cv2.imwrite(os.path.join(tf_flags.res_dir, "c_"+fname), (center*25).astype(np.uint8))
                    # from scipy.io import savemat
                    # savemat(os.path.join(tf_flags.res_dir, fname[:-3]+'mat'), dict(emb=emb))

                except Exception as exc:
                    print(exc)
                    csvfile.close()
                    print(exc)
                    print(e.score())
                    break


if __name__ == '__main__':

    tf.app.flags.DEFINE_string("phase", "train",
                               "model phase: train/test/sparsity_tune.")

    # architecture config
    tf.app.flags.DEFINE_string("architecture", "d9",
                               "architecture of the backbone network, d7/d9")
    tf.app.flags.DEFINE_boolean("include_bg", True,
                                "whether include background as an independent object")
    tf.app.flags.DEFINE_integer("embedding_dim", 8,
                                "dimension of the embedding")

    # training config
    tf.app.flags.DEFINE_string("train_dir", "./tfrecords/U2OScell/train",
                               "dataset directory")
    tf.app.flags.DEFINE_string("val_dir", "./tfrecords/U2OScell/val",
                               "dataset directory")
    tf.app.flags.DEFINE_string("image_depth", "uint8",
                               "depth of image: uint8/uint16")
    tf.app.flags.DEFINE_integer("image_channels", 1, "number of image channels")            
    tf.app.flags.DEFINE_string("model_dir", "./model_U2OScell",
                               "checkpoint and summary directory.")
    tf.app.flags.DEFINE_string("checkpoint_prefix", "model",
                               "checkpoint name for restoring.")
    tf.app.flags.DEFINE_float("lr", 0.0001,
                              "Learning Rate.")
    tf.app.flags.DEFINE_integer("batch_size", 4,
                                "batch size for training.")
    tf.app.flags.DEFINE_integer("training_epoches", 500,
                                "total training steps.")
    tf.app.flags.DEFINE_integer("summary_steps", 100,
                                "summary period.")
    tf.app.flags.DEFINE_integer("save_steps", 2000,
                                "checkpoint period.")
    tf.app.flags.DEFINE_integer("validation_steps", 200,
                                "validation period.")

    # test config
    tf.app.flags.DEFINE_string("res_dir", "./model_orthogonal",
                               "result directory")
    tf.app.flags.DEFINE_boolean("keep_size", False,
                                "resize to original size or not when testing")
    
    tf.app.run(main=main)
