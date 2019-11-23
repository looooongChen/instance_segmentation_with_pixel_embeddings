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
        
        # # load dataset from tfrecords
        # tf_dir = "./tfrecords/U2OScell/val"
        # val_tf = [os.path.join(tf_dir, f) for f in os.listdir(tf_dir)]
        # # build dataset
        # val_ds = tf.data.TFRecordDataset(val_tf)
        # preprocess_f = lambda sample: extract_fn(sample, 
        #                                          image_channels=tf_flags.image_channels,
        #                                          image_depth=tf_flags.image_depth,
        #                                          dist_map=tf_flags.dist_branch)
        # val_ds = val_ds.map(preprocess_f).batch(1)
        # val_iterator = val_ds.make_one_shot_iterator()
        # val_example = val_iterator.get_next()
        # with tf.Session() as sess:
        #     model = LocalDisNet(sess, tf_flags)
        #     model.restore_model()
        #     count = 1
        #     while True:
        #         print(count)
        #         sample = sess.run(val_example)
        #         img = np.squeeze(sample['image/image'])
        #         segs, embs = model.segment_from_seed([img], seed_thres=0.7, similarity_thres=0.7, resize=False)
        #         e = embs[0,:,:,0:3]
        #         e = 255*(e-e.min())/(e.max()-e.min())
        #         img =(img-img.min())/(img.max()-img.min())*255
        #         imsave(os.path.join(tf_flags.test_res, 'img_'+str(count)+'.png'), img.astype(np.uint8))
        #         imsave(os.path.join(tf_flags.test_res, 'embedding_'+str(count)+'.png'), e.astype(np.uint8))
        #         save_indexed_png(os.path.join(tf_flags.test_res, 'seg_'+str(count)+'.png'), segs[0].astype(np.uint8))
        #         count += 1
        #         if count == 10:
        #             break


        img_path = {f: os.path.join(tf_flags.test_dir, f) for f in os.listdir(tf_flags.test_dir)}

        if not os.path.exists(tf_flags.test_res):
            os.makedirs(tf_flags.test_res)

        with tf.Session() as sess:
            model = LocalDisNet(sess, tf_flags)
            model.restore_model()
            for f_name, f_path in img_path.items():
                img = imread(f_path)
                segs, embs = model.segment_from_seed([img], seed_thres=0.7, similarity_thres=0.7, resize=False)
                e = embs[0,:,:,0:3]
                e = 255*(e-e.min())/(e.max()-e.min())
                img =(img-img.min())/(img.max()-img.min())*255
                imsave(os.path.join(tf_flags.test_res, os.path.splitext(f_name)[0]+'_img.png'), img.astype(np.uint8))
                imsave(os.path.join(tf_flags.test_res, os.path.splitext(f_name)[0]+'_emb.png'), e.astype(np.uint8))
                save_indexed_png(os.path.join(tf_flags.test_res, os.path.splitext(f_name)[0]+'_seg.png'), segs[0].astype(np.uint8))

    elif tf_flags.phase == 'evaluation':
        e = Evaluator(gt_type="mask")
        # implement your the evaluation based on your dataset with Evaluator
        pass

if __name__ == '__main__':

    tf.app.flags.DEFINE_string("phase", "train",
                               "model phase: train/test/evaluation")

    # architecture config
    tf.app.flags.DEFINE_string("architecture", "d9",
                               "backbone network, d9 for standard U-Net, d7 for simplified U-Net")
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
    tf.app.flags.DEFINE_string("test_dir", "./CVPPP_test/A4_test",
                               "evaluation dataset directory")
    tf.app.flags.DEFINE_string("test_res", "./CVPPP_test/A4_res",
                               "evaluation dataset directory")
    
    tf.app.run(main=main)
