import dan_model
import dan_run_loop

import os
import sys
import glob

import numpy as np
import cv2
import tensorflow as tf
import shutil
from distutils.dir_util import copy_tree


resume_path = '/model'

if os.path.exists(resume_path):
    # shutil.copytree(resume_path, '/output')
    copy_tree(resume_path, '/output')



class VGG16Model(dan_model.Model):
    def __init__(self,num_lmark,data_format=None):
        
        img_size=112
        filter_sizes=[64,128,256,512]
        num_convs=2
        kernel_size=3

        super(VGG16Model,self).__init__(
            num_lmark=num_lmark,
            img_size=img_size,
            filter_sizes=filter_sizes,
            num_convs=num_convs,
            kernel_size=kernel_size,
            data_format=data_format
        )

def get_filenames(data_dir):
    listext = ['*.png','*.jpg']

    imagelist = []
    for ext in listext:
        p = os.path.join(data_dir, ext)
        imagelist.extend(glob.glob(p))

    ptslist = []
    for image in imagelist:
        ptslist.append(os.path.splitext(image)[0] + ".ptv")

    return imagelist, ptslist


def get_synth_input_fn():
    return dan_run_loop.get_synth_input_fn(112, 112, 1, 74.)

def vgg16_input_fn(is_training,data_dir,batch_size=64,num_epochs=1,num_parallel_calls=1, multi_gpu=False):
    img_path,pts_path = get_filenames(data_dir)

    def decode_img_pts(img,pts,is_training):
        img = cv2.imread(img.decode(), cv2.IMREAD_GRAYSCALE)
        pts = np.loadtxt(pts.decode(),dtype=np.float32,delimiter=',')
        return img[:,:,np.newaxis].astype(np.float32),pts.astype(np.float32)

    map_func=lambda img,pts,is_training:tuple(tf.py_func(decode_img_pts,[img,pts,is_training],[tf.float32,tf.float32]))

    img = tf.data.Dataset.from_tensor_slices(img_path)
    pts = tf.data.Dataset.from_tensor_slices(pts_path)

    dataset = tf.data.Dataset.zip((img, pts))
    num_images = len(img_path)

    return dan_run_loop.process_record_dataset(dataset,is_training,batch_size,
                                               num_images,map_func,num_epochs,num_parallel_calls,
                                               examples_per_epoch=num_images, multi_gpu=multi_gpu)

def read_dataset_info(data_dir):
    mean_shape = np.loadtxt(os.path.join(data_dir,'mean_shape.ptv'),dtype=np.float32,delimiter=',')
    imgs_mean = np.loadtxt(os.path.join(data_dir,'imgs_mean.ptv'),dtype=np.float32,delimiter=',')
    imgs_std = np.loadtxt(os.path.join(data_dir,'imgs_std.ptv'),dtype=np.float32,delimiter=',')
    return mean_shape.astype(np.float32) ,imgs_mean.astype(np.float32),imgs_std.astype(np.float32)


def img_input_fn( img_path, rect, img_size, num_lmark):


    def _get_frame():

        frame = cv2.imread(img_path)

        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # rect = (np.round(rect)).astype(np.int32)
        # frame = frame[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
        frame = cv2.resize(frame,(img_size,img_size), interpolation=cv2.INTER_CUBIC).astype(np.float32)

        # imgs_mean_ = np.loadtxt('/home/morzh/temp/imgs_mean.ptv', delimiter=',')
        # imgs_std__ = np.loadtxt('/home/morzh/temp/imgs_std.ptv', delimiter=',')

        # cv2.imshow("asdva", frame)
        # cv2.waitKey(-1)

        # frame = frame - imgs_mean_
        # frame = frame / imgs_std__

        yield (frame,np.zeros([num_lmark,2],np.float32))


    def input_fn():
        dataset = tf.data.Dataset.from_generator(_get_frame,(tf.float32,tf.float32), (tf.TensorShape([img_size,img_size]),tf.TensorShape([num_lmark,2])))
        return dataset

    return input_fn


def video_input_fn(data_dir,img_size,num_lmark):
    video = cv2.VideoCapture(data_dir)

    def _get_frame():
        while True:
            _,frame = video.read()
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame,(img_size,img_size)).astype(np.float32)
            yield (frame, np.zeros([num_lmark,2],np.float32))

    def input_fn():
        dataset = tf.data.Dataset.from_generator(_get_frame,(tf.float32,tf.float32),(tf.TensorShape([img_size,img_size]),tf.TensorShape([num_lmark,2])))
        return dataset

    return input_fn





def main(argv):
    parser = dan_run_loop.DANArgParser()
    parser.set_defaults(data_dir='./data_dir',
                        model_dir='/output',
                        data_format='channels_last',
                        train_epochs=20,
                        epochs_per_eval=10,
                        batch_size=64)

    flags = parser.parse_args(args=argv[1:])


    mean_shape = None
    imgs_mean = None
    imgs_std = None

    flags_trans = { 'train':tf.estimator.ModeKeys.TRAIN, 'eval':tf.estimator.ModeKeys.EVAL, 'predict':tf.estimator.ModeKeys.PREDICT }

    flags.mode = flags_trans[flags.mode]

    if flags.mode == tf.estimator.ModeKeys.TRAIN:
        mean_shape,imgs_mean,imgs_std = read_dataset_info(flags.data_dir)

    def vgg16_model_fn(features, labels, mode, params):
        return dan_run_loop.dan_model_fn(features=features,
                            groundtruth=labels,
                            mode=mode,
                            stage=params['dan_stage'],                                                    
                            num_lmark=params['num_lmark'],
                            model_class=VGG16Model,
                            mean_shape=mean_shape,
                            imgs_mean=imgs_mean,
                            imgs_std=imgs_std,
                            data_format=params['data_format'],
                            multi_gpu=params['multi_gpu'])

    input_function = flags.use_synthetic_data and get_synth_input_fn() or vgg16_input_fn

    if flags.mode == tf.estimator.ModeKeys.PREDICT:

        faceset = '/media/morzh/ext4_volume/data/Faces/all_in_one/set_004_rect/'
        file_img  = '2UiNSKC3sGw.jpg'
        file_rect = faceset+file_img+'.rect'
        # rect = np.loadtxt(file_rect)

        input_function = img_input_fn( faceset+file_img,  (0,0,112,112), 112, 74)

    dan_run_loop.dan_main(flags, vgg16_model_fn, input_function)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(argv=sys.argv)
