""" Utility functions. """
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

## Image helper
def path_process(image_path):
    depth_path = image_path.replace('_scene.jpg', '_depth1D.jpg')
    box_path = image_path.replace('.jpg', '.dat')
    return depth_path, box_path


def get_images(paths, nb_samples=None, shuffle=False, whole=False):
    '''
    :param paths:  paths of the fine-grained face folders
    :param nb_samples: for each fine-grained face folder, sample how many faces
    :param shuffle: no use
    :param whole: no use
    :return: sampled facial images
    '''
    label_images = []

    for i, path in enumerate(paths):
        files = os.listdir(path)
        images = [image for image in files if image.endswith('_scene.jpg')]
        if len(images) < nb_samples:
            raise ValueError('please check that whether each class contains enough images for the support set,'
                             'the class path is :  ' + path)

        else:
            sampled_images = random.sample(images, nb_samples)
            for i, image in enumerate(sampled_images):
                face_path = os.path.join(path, image)
                label_images.append(face_path)

    return label_images


def get_images_specify(paths, sub_dir='', nb_samples=5, shuffle=False, whole=False):
    label_images = []
    for i, path in enumerate(paths):
        sub_path = os.path.join(path, sub_dir)

        files = os.listdir(sub_path)
        images = [image for image in files if image.endswith('jpg')]
        if len(images) < nb_samples:
            raise ValueError('please check that whether each class contains enough images for the support set,'
                             'the class path is :  ' + sub_path)
        sampled_images = random.sample(images, nb_samples)
        for image in sampled_images:
            face_path = os.path.join(sub_path, image)
            label_images.append(face_path)

    return label_images


def crop_face_from_scene(image,face_name_full, scale=1.2):
    '''
    :param image: facial image array
    :param face_name_full: facial box file path
    :param scale:  the size scale to crop the face region from the facial image
    :return: the cropped facial region
    '''
    f=open(face_name_full,'r')
    lines=f.readlines()
    y1,x1,w,h=[float(ele) for ele in lines[:4]]
    f.close()
    y2=y1+w
    x2=x1+h

    y_mid=(y1+y2)/2.0
    x_mid=(x1+x2)/2.0
    shape =image.shape
    if len(shape) == 2:
        h_img, w_img = shape
    else:
        h_img, w_img, channels = shape
    w_scale=scale*w
    h_scale=scale*h
    y1=y_mid-w_scale/2.0
    x1=x_mid-h_scale/2.0
    y2=y_mid+w_scale/2.0
    x2=x_mid+h_scale/2.0
    y1=max(y1,0.0)
    x1=max(x1,0.0)
    y2=min(y2,float(w_img))
    x2=min(x2,float(h_img))

    x1 = int(x1)
    x2 = int(x2)
    y1 = int(y1)
    y2 = int(y2)

    region=image[x1:x2, y1:y2]

    return region


def contrast_depth_conv(input, dilation_rate=1, op_name='contrast_depth'):
    ''' compute contrast depth in both of (out, label) '''
    assert (input.get_shape()[-1] == 1)

    kernel_filter_list = [
        [[1, 0, 0], [0, -1, 0], [0, 0, 0]], [[0, 1, 0], [0, -1, 0], [0, 0, 0]], [[0, 0, 1], [0, -1, 0], [0, 0, 0]],
        [[0, 0, 0], [1, -1, 0], [0, 0, 0]], [[0, 0, 0], [0, -1, 1], [0, 0, 0]],
        [[0, 0, 0], [0, -1, 0], [1, 0, 0]], [[0, 0, 0], [0, -1, 0], [0, 1, 0]], [[0, 0, 0], [0, -1, 0], [0, 0, 1]]
    ]
    kernel_filter = np.array(kernel_filter_list, np.float32)
    kernel_filter = np.expand_dims(kernel_filter, axis=2)
    kernel_filter_tf = tf.constant(kernel_filter, dtype=tf.float32)

    if dilation_rate == 1:
        contrast_depth = tf.nn.conv2d(input, kernel_filter_tf, strides=[1, 1, 1, 1], padding='SAME', name=op_name)
    else:
        contrast_depth = tf.nn.atrous_conv2d(input, kernel_filter_tf,rate=dilation_rate, padding='SAME', name=op_name)

    return contrast_depth


def CDL(out,label):
    loss1 = contrast_depth_loss(out, label)
    loss2 = L2_loss(out, label)
    return loss1 + loss2


def contrast_depth_loss(out, label):
    '''
    compute contrast depth in both of (out, label),
    then get the loss of them
    tf.atrous_convd match tf-versions: 1.4
    '''
    contrast_out = contrast_depth_conv(out, 1, 'contrast_out')
    contrast_label = contrast_depth_conv(label, 1, 'contrast_label')

    loss = tf.pow(contrast_out - contrast_label, 2)
    loss = tf.reduce_mean(loss)

    return loss


def L2_loss(out, label):
    loss = tf.pow(out - label, 2)

    #loss = tf.square(loss*loss)
    loss = tf.reduce_mean(loss)
    return loss
    #return tf.sqrt(loss)


from tensorflow.python.training.moving_averages import assign_moving_average
def batch_norm(x, training=True, eps=1e-05, decay=0.9, affine=True, name=None):
    with tf.variable_scope(name, default_name='BatchNorm2d'):
        params_shape = x.get_shape().as_list()[-1:]

        axis = [k for k in range(len(x.get_shape().as_list()) - 1)]
        mean, variance = tf.nn.moments(x, axis, name='moments')

        if affine:
            beta = tf.get_variable('beta', params_shape,
                                   initializer=tf.zeros_initializer)
            gamma = tf.get_variable('gamma', params_shape,
                                    initializer=tf.ones_initializer)
            x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
        else:
            x = tf.nn.batch_normalization(x, mean, variance, None, None, eps)
        return x
