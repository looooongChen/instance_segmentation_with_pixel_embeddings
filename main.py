import os
import tensorflow as tf
import numpy as np
from Net import LocalDisNet
from skimage.io import imread, imsave
from preprocess import extract_fn
from utils.img_io import save_indexed_png
import cv2
from utils.evaluation import Evaluator


def main(_):
    tf_flags = tf.app.flags.FLAGS

    if tf_flags.phase == "train":
        with tf.Session() as sess:
            model = LocalDisNet(sess, tf_flags)
            if tf_flags.validation:
                val_dir = os.path.join(tf_flags.val_dir)
            else:
                val_dir = None
            model.train(tf_flags.batch_size,
                        tf_flags.training_epoches,
                        os.path.join(tf_flags.train_dir),
                        val_dir)
    elif tf_flags.phase == 'prediction':

        if not os.path.exists(tf_flags.test_res):
            os.makedirs(tf_flags.test_res)

        img_path = {f: os.path.join(tf_flags.test_dir, f) for f in os.listdir(tf_flags.test_dir)}

        if not os.path.exists(tf_flags.test_res):
            os.makedirs(tf_flags.test_res)

        with tf.Session() as sess:
            model = LocalDisNet(sess, tf_flags)
            model.restore_model()
            for f_name, f_path in img_path.items():
                img = imread(f_path)
                print("Processing: ", f_path)
                segs = model.segment_from_seed([img], seed_thres=0.7, similarity_thres=0.7, resize=True)
                save_indexed_png(os.path.join(tf_flags.test_res, os.path.splitext(f_name)[0]+'_seg.png'), segs[0].astype(np.uint8))

    elif tf_flags.phase == 'evaluation':
        e = Evaluator(gt_type="mask")
        # implement your the evaluation based on your dataset with Evaluator
        pass

if __name__ == '__main__':

    tf.app.flags.DEFINE_string("phase", "train",
                               "model phase: train/test/evaluation")

    # architecture config
    tf.app.flags.DEFINE_boolean("dist_branch", True,
                                "whether train dist regression branch or not")
    tf.app.flags.DEFINE_boolean("include_bg", True,
                                "whether include background as an independent object")
    tf.app.flags.DEFINE_integer("embedding_dim", 16,
                                "dimension of the embedding")

    # training config
    tf.app.flags.DEFINE_string("train_dir", "./tfrecords/U2OScell/train",
                               "train dataset directory")
    tf.app.flags.DEFINE_boolean("validation", True,
                                "run validation during training or not, if False, --val_dir will be ignored")
    tf.app.flags.DEFINE_string("val_dir", "./tfrecords/U2OScell/val",
                               "validation dataset directory")
    tf.app.flags.DEFINE_string("image_depth", "uint16",
                               "depth of image: uint8/uint16")
    tf.app.flags.DEFINE_integer("image_channels", 3, "number of image channels")            
    tf.app.flags.DEFINE_string("model_dir", "./model_CVPPP2017",
                               "checkpoint and summary directory.")
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
    tf.app.flags.DEFINE_string("test_dir", "./test/cvppp_test",
                               "evaluation dataset directory")
    tf.app.flags.DEFINE_string("test_res", "./test/cvppp_res",
                               "evaluation dataset directory")
    
    tf.app.run(main=main)
