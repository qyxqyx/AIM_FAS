""" Code for loading data. """
import numpy as np
import cv2
import random
import tensorflow as tf
from tensorflow.python.platform import flags
from utils import get_images, path_process, crop_face_from_scene
import os
import copy

FLAGS = flags.FLAGS

class Task_dataset(object):
    def __init__(self):
        metatrain_folder = FLAGS.data_path + '/Train'
        if FLAGS.test_set:
            metaval_folder = FLAGS.data_path + '/Test'
        else:
            metaval_folder = FLAGS.data_path + '/Dev'

        metatrain_folders = [os.path.join(metatrain_folder, label) \
            for label in os.listdir(metatrain_folder) \
            if os.path.isdir(os.path.join(metatrain_folder, label)) \
            ]
        # get the positive and negative folder if num_classes is 2
        self.metatrain_folders_p = [folder for folder in metatrain_folders if folder.endswith('_1')]
        self.metatrain_folders_n = [folder for folder in metatrain_folders if not folder.endswith('_1')]

        metaval_folders = [os.path.join(metaval_folder, label) \
            for label in os.listdir(metaval_folder) \
            if os.path.isdir(os.path.join(metaval_folder, label)) \
            ]
        # get the positive and negative folder if num_classes is 2

        self.metaval_folders_p = [folder for folder in metaval_folders if folder.endswith('_1')]
        self.metaval_folders_n = [folder for folder in metaval_folders if not folder.endswith('_1')]

        self.metatrain_character_folders = metatrain_folders
        self.metaval_character_folders = metaval_folders

        self.num_total_train_batches = FLAGS.num_train_tasks
        self.num_total_val_batches = FLAGS.num_test_tasks

        if FLAGS.train:
            self.store_data_per_task(train=True)
        self.store_data_per_task(train=False)

        self.val_task_pointer = 0

        self.image_lists = []
        self.out_faces = []
        self.out_depthes = []
        self.iterators = []
        for i in range(FLAGS.meta_batch_size*FLAGS.num_gpus):
            image_list = tf.placeholder(dtype=tf.string, shape=[None, ])
            self.image_lists.append(image_list)
            dataset = tf.data.Dataset.from_tensor_slices(image_list)
            dataset = dataset.map(self.read_image, num_parallel_calls=24)
            dataset = dataset.batch(200)
            iterator = dataset.make_initializable_iterator()
            one_element = iterator.get_next()
            [face, depth] = one_element
            face = tf.expand_dims(face, axis=0)
            depth = tf.expand_dims(depth, axis=0)
            self.out_faces.append(face)
            self.out_depthes.append(depth)
            self.iterators.append(iterator.initializer)
        self.out_faces = tf.concat(self.out_faces, axis=0)
        self.out_depthes = tf.concat(self.out_depthes, axis=0)


    def store_data_per_task(self, train=True):
        if train:
            self.train_tasks_data_classes = []
            for i in range(self.num_total_train_batches):
                s_p_folder = random.sample(self.metatrain_folders_p, 1)
                s_n_folder = random.sample(self.metatrain_folders_n, 1)
                s_task_folders = s_p_folder + s_n_folder

                shot_list = list(FLAGS.shot_list.split(','))
                support_num = int(random.choice(shot_list))
                random.shuffle(s_task_folders)
                support_images = get_images(s_task_folders, nb_samples=FLAGS.num_support-support_num )

                q_p_folder = random.sample(self.metatrain_folders_p, 1)
                q_n_folder = random.sample(self.metatrain_folders_n, 1)
                q_task_folders = q_p_folder + q_n_folder

                random.shuffle(q_task_folders)
                query_images = get_images(q_task_folders, nb_samples=FLAGS.num_query_t + support_num)

                support_add = [query_images[id] for id in range(len(query_images)) if
                               id % (FLAGS.num_query_t + support_num) < support_num]
                query = [query_images[id] for id in range(len(query_images)) if
                         id % (FLAGS.num_query_t + support_num) >= support_num]
                support_images.extend(support_add)

                data_class_task = Files_per_task(support_images, query, i)
                self.train_tasks_data_classes.append(data_class_task)
        else:
            self.val_tasks_data_classes = []
            for i in range(self.num_total_val_batches):
                if FLAGS.test_method == 0:
                    p_folder = random.sample(self.metaval_folders_p, 1)
                    n_folder = random.sample(self.metaval_folders_n, 1)
                    task_folders = p_folder + n_folder

                    random.shuffle(task_folders)
                    support_images = get_images(task_folders, nb_samples=FLAGS.num_support - FLAGS.num_shot)

                else:
                    p_folder = random.sample(self.metatrain_folders_p, 1)
                    n_folder = random.sample(self.metatrain_folders_n, 1)
                    task_folders = p_folder + n_folder

                    random.shuffle(task_folders)
                    support_images = get_images(task_folders, nb_samples=FLAGS.num_support - FLAGS.num_shot)

                p_folder = random.sample(self.metaval_folders_p, 1)
                n_folder = random.sample(self.metaval_folders_n, 1)
                task_folders = p_folder + n_folder

                random.shuffle(task_folders)
                sampled = get_images_specify(task_folders, nb_samples=FLAGS.num_shot+FLAGS.num_query_v)
                support_add = [sampled[id] for id in range(len(sampled)) if
                               id % (FLAGS.num_shot + FLAGS.num_query_v) < FLAGS.num_shot]
                query       = [sampled[id] for id in range(len(sampled)) if
                               id % (FLAGS.num_shot + FLAGS.num_query_v) >= FLAGS.num_shot]

                support_images.extend(support_add)

                data_class_task = Files_per_task(support_images, query, i)
                self.val_tasks_data_classes.append(data_class_task)


    def read_data_per_tesk(self,task_index, train=True):
        if train:
            task_class = copy.deepcopy(self.train_tasks_data_classes[task_index])
        else:
            task_class = copy.deepcopy(self.val_tasks_data_classes[task_index])

        support_images = task_class.support_images
        query_images = task_class.query_images

        random.shuffle(support_images)
        random.shuffle(query_images)

        return support_images, query_images


    def get_data_n_tasks(self, meta_batch_size, train=True):
        if train:
            task_indexes = np.random.choice(self.num_total_train_batches, meta_batch_size)
        else:
            if meta_batch_size + self.val_task_pointer >= self.num_total_val_batches:
                task_indexes = np.arange(self.val_task_pointer, self.val_task_pointer + meta_batch_size)
                self.val_task_pointer = 0
            else:
                task_indexes = np.arange(self.val_task_pointer, self.val_task_pointer + meta_batch_size)
                self.val_task_pointer += meta_batch_size

        train_files, test_files = [], []

        for task_index in task_indexes:
            task_train_files, task_test_files = self.read_data_per_tesk(task_index, train)

            train_files.append(task_train_files)
            test_files.append(task_test_files)

        return train_files, test_files


    def _parser(self, image_path):
        '''
        :param image_path: image path
        :return: the cropped face and the facial depth
        '''
        image_path = image_path.decode()
        image = cv2.imread(image_path)
        depth_path, box_path = path_process(image_path)
        face = crop_face_from_scene(image, box_path)
        face = cv2.resize(face, (256, 256))
        if '_1/' in image_path:
            # living face
            depth = cv2.imread(depth_path, 0)
            face_depth = crop_face_from_scene(depth, box_path)
            face_depth = cv2.resize(face_depth, (32, 32))
            depth = face_depth[:, :, np.newaxis]
        else:
            # spoofing face
            depth = np.zeros(shape=[32, 32, 1])
        face = face.astype(np.float32) - 127.5
        depth = depth.astype(np.float32) / 256
        return face, depth


    def read_image(self, face_path):
        face, depth = tf.py_func(self._parser, inp=[face_path], Tout=[tf.float32, tf.float32])
        return face, depth


def make_one_hot(data, classes):
    return (np.arange(classes)==data[:,None]).astype(np.integer)


class Files_per_task(object):
    def __init__(self, support_images, query_images, task_index):
        self.support_images = support_images
        self.query_images = query_images
        self.task_index = task_index


