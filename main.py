import numpy as np
import tensorflow as tf
import datetime

from model import AIM_FAS, get_err_threhold, performances
from tensorflow.python.platform import flags
from Task_Generator import Task_dataset
from tensorflow.python import pywrap_tensorflow

FLAGS = flags.FLAGS

flags.DEFINE_integer('metatrain_iterations', 20000, 'number of meta-training iterations.')
flags.DEFINE_integer('num_classes', 2, 'number of classes.')
flags.DEFINE_integer('meta_batch_size', 1, 'number of tasks sampled for meta-train per gpu')
flags.DEFINE_float('meta_lr', 0.0001, 'the base learning rate of the meta-learner')
flags.DEFINE_float('update_lr', 0.001, 'inner update lr')
flags.DEFINE_integer('num_support', 15, 'number of examples used for inner update.')
flags.DEFINE_integer('num_shot', 0,     'number of images that are belong to the same class with query when testing.')
flags.DEFINE_integer('num_query_t', 15, 'number of examples of each class in query set of each training task.')
flags.DEFINE_integer('num_query_v', 15, 'number of examples of each class in query set of each validation task.')
flags.DEFINE_integer('num_updates', 3,  'number of inner gradient updates during training.')
flags.DEFINE_integer('test_num_updates', 20, 'number of inner gradient updates during testing')

flags.DEFINE_integer('num_train_tasks', 1000, 'number of meta training tasks.')
flags.DEFINE_integer('num_test_tasks', 20, 'number of meta training tasks.')
flags.DEFINE_integer('lr_decay_itr', 0, 'number of iteration that decay the meta lr')
flags.DEFINE_float('l2_alpha', 0.00001, 'param of the l2_norm loss')
flags.DEFINE_float('l1_alpha', 0.00, 'param of the l1_norm loss')

flags.DEFINE_integer('base_num_filters', 16, '')
flags.DEFINE_string('loss', 'L2', 'L2 or Con')

flags.DEFINE_integer('num_gpus', 8, 'multi-gpus')
flags.DEFINE_string('shot_list', '0,1,3,5,7,9', '')
flags.DEFINE_string('inner_losses', '1, -1', 'which inner update step loss is used to train the meta-learner')

flags.DEFINE_bool('restore', False, '')
flags.DEFINE_integer('test_method', 1, 'test method, 0:support of test task are collected on test set; 1:on the train set')

## Logging, saving, and testing options
flags.DEFINE_string('logdir', 'logs/FAS/', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', False, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_string('pretrain_model', 'models/model29000', 'the path of the pretrained model')
flags.DEFINE_string('data_path', 'dataset/OULU-ZF', 'path of the dataset.')


NUM_TEST_POINTS = int(FLAGS.num_test_tasks/FLAGS.meta_batch_size/FLAGS.num_gpus)

def train(model, saver, sess, hyper_setting, task_generator, resume_itr=0):
    PRINT_INTERVAL = 20
    TEST_PRINT_INTERVAL = 100

    tf.summary.FileWriter(FLAGS.logdir + '/' + hyper_setting, sess.graph)
    print(hyper_setting)
    print('Done initializing, starting training.')
    prelosses, postlosses = [], []

    min_APCER = APCER_95 = min_APCER_itr = 1
    min_NPCER = NPCER_95 = min_NPCER_itr = 1
    min_ACER = ACER_95 = min_ACER_itr = 1
    min_ACER_pre = ACER_95_pre = min_ACER_pre_itr = 1

    postlosses2way_APCER, postlosses2way_NPCER, postlosses2way_ACER = [], [], []

    for itr in range(resume_itr, FLAGS.metatrain_iterations):
        # 调节learning rate
        if FLAGS.lr_decay_itr > 0:
            if int(itr/FLAGS.lr_decay_itr) == 0:
                lr1 = FLAGS.meta_lr
            elif int(itr/FLAGS.lr_decay_itr) == 1:
                lr1 = FLAGS.meta_lr/10
            else:
                lr1 = FLAGS.meta_lr/100

            if int(itr % FLAGS.lr_decay_itr) < 2:
                print('change the mata lr to:' + str(lr1) + ', ----------------------------')
        else:
            lr1 = FLAGS.meta_lr

        feed_dict = {model.meta_lr: lr1}
        feed_dict_data = {}

        if itr == resume_itr:
            meta_train_files, meta_test_files = task_generator.get_data_n_tasks(FLAGS.num_gpus * FLAGS.meta_batch_size, train=True)
            for task_id in range(FLAGS.meta_batch_size*FLAGS.num_gpus):
                im_file = meta_train_files[task_id]
                im_file_test = meta_test_files[task_id]
                im_file.extend(im_file_test)
                feed_dict_data[task_generator.image_lists[task_id]] = im_file
            sess.run(task_generator.iterators, feed_dict=feed_dict_data)
            [meta_ims, meta_depthes] = sess.run([task_generator.out_faces, task_generator.out_depthes])

            meta_train_ims = meta_ims[:, :FLAGS.num_classes * FLAGS.num_support, :]
            meta_test_ims = meta_ims[:, FLAGS.num_classes * FLAGS.num_support:, :]
            meta_train_lbls = meta_depthes[:, :FLAGS.num_classes * FLAGS.num_support, :]
            meta_test_lbls = meta_depthes[:, FLAGS.num_classes * FLAGS.num_support:, :]

        feed_dict[model.inputa] = meta_train_ims
        feed_dict[model.inputb] = meta_test_ims
        feed_dict[model.labela] = meta_train_lbls
        feed_dict[model.labelb] = meta_test_lbls

        meta_train_files, meta_test_files = task_generator.get_data_n_tasks(FLAGS.num_gpus*FLAGS.meta_batch_size, train=True)
        for task_id in range(FLAGS.meta_batch_size * FLAGS.num_gpus):
            im_file = meta_train_files[task_id]
            im_file_test = meta_test_files[task_id]
            im_file.extend(im_file_test)
            feed_dict_data[task_generator.image_lists[task_id]] = im_file
        sess.run(task_generator.iterators, feed_dict=feed_dict_data)

        input_tensors = [model.metatrain_op]

        input_tensors.extend([model.total_loss1, model.total_losses2[FLAGS.num_updates-1]])

        input_tensors.extend(
            [model.APCER[FLAGS.num_updates-1], 
             model.NPCER[FLAGS.num_updates-1],
             model.ACER[FLAGS.num_updates-1],
             task_generator.out_faces, 
             task_generator.out_depthes])

        result = sess.run(input_tensors, feed_dict)

        prelosses.append(result[1])
        postlosses.append(result[2])
        postlosses2way_APCER.append(result[3])
        postlosses2way_NPCER.append(result[4])
        postlosses2way_ACER.append(result[5])

        meta_ims = result[6]
        meta_depthes = result[7]
        meta_train_ims = meta_ims[:, :FLAGS.num_classes * FLAGS.num_support, :]
        meta_test_ims = meta_ims[:, FLAGS.num_classes * FLAGS.num_support:, :]
        meta_train_lbls = meta_depthes[:, :FLAGS.num_classes * FLAGS.num_support, :]
        meta_test_lbls = meta_depthes[:, FLAGS.num_classes * FLAGS.num_support:, :]

        if (itr!=0) and itr % PRINT_INTERVAL == 0:
            print_str = 'Iteration ' + str(itr)
            print_str += ': ' + str(np.mean(prelosses)) + ', ' + str(np.mean(postlosses))

            print_str += ', ' + str(np.mean(postlosses2way_APCER)) \
                         + ', ' + str(np.mean(postlosses2way_NPCER)) \
                         + ', ' + str(np.mean(postlosses2way_ACER)) 
            print(str(datetime.datetime.now())[:-7], print_str)
            prelosses, postlosses = [], []

            postlosses2way_APCER, postlosses2way_NPCER, postlosses2way_ACER = [], [], []

        if (itr!=0) and itr % TEST_PRINT_INTERVAL == 0:
            metaval_accuracies = []
            for test_itr in range(NUM_TEST_POINTS):
                feed_dict_data_test = {}
                feed_dict_test = {model.meta_lr: 0}
                if test_itr == 0:
                    metaval_train_files, metaval_test_files = task_generator.get_data_n_tasks(
                        FLAGS.num_gpus * FLAGS.meta_batch_size, train=False)
                    for task_id in range(FLAGS.meta_batch_size * FLAGS.num_gpus):
                        im_file = metaval_train_files[task_id]
                        im_file_test = metaval_test_files[task_id]
                        im_file.extend(im_file_test)
                        feed_dict_data_test[task_generator.image_lists[task_id]] = im_file
                    sess.run(task_generator.iterators, feed_dict=feed_dict_data_test)
                    [metaval_ims, metaval_depthes] = sess.run([task_generator.out_faces, task_generator.out_depthes])

                    metaval_train_ims = metaval_ims[:, :FLAGS.num_classes * FLAGS.num_support, :]
                    metaval_test_ims = metaval_ims[:, FLAGS.num_classes * FLAGS.num_support:, :]
                    metaval_train_lbls = metaval_depthes[:, :FLAGS.num_classes * FLAGS.num_support, :]
                    metaval_test_lbls = metaval_depthes[:, FLAGS.num_classes * FLAGS.num_support:, :]

                feed_dict_test[model.inputa] = metaval_train_ims
                feed_dict_test[model.inputb] = metaval_test_ims
                feed_dict_test[model.labela] = metaval_train_lbls
                feed_dict_test[model.labelb] = metaval_test_lbls

                metaval_train_files, metaval_test_files = task_generator.get_data_n_tasks(
                    FLAGS.num_gpus * FLAGS.meta_batch_size, train=False)
                for task_id in range(FLAGS.meta_batch_size * FLAGS.num_gpus):
                    im_file = metaval_train_files[task_id]
                    im_file_test = metaval_test_files[task_id]
                    im_file.extend(im_file_test)
                    feed_dict_data_test[task_generator.image_lists[task_id]] = im_file
                sess.run(task_generator.iterators, feed_dict=feed_dict_data_test)

                input_tensors = [[model.metaval_total_loss1] + model.metaval_total_losses2 +
                                 model.metaval_APCER + model.metaval_NPCER + model.metaval_ACER,
                                 task_generator.out_faces, task_generator.out_depthes]

                result = sess.run(input_tensors, feed_dict_test)
                metaval_accuracies.append(result[0])
                metaval_ims = result[-2]
                metaval_depthes = result[-1]
                metaval_train_ims = metaval_ims[:, :FLAGS.num_classes * FLAGS.num_support, :]
                metaval_test_ims = metaval_ims[:, FLAGS.num_classes * FLAGS.num_support:, :]
                metaval_train_lbls = metaval_depthes[:, :FLAGS.num_classes * FLAGS.num_support, :]
                metaval_test_lbls = metaval_depthes[:, FLAGS.num_classes * FLAGS.num_support:, :]

            metaval_accuracies = np.array(metaval_accuracies)
            means = np.mean(metaval_accuracies, 0)
            stds = np.std(metaval_accuracies, 0)
            ci95 = 1.96 * stds / np.sqrt(NUM_TEST_POINTS)
            print('----------------------------------------', itr)
            print('Mean validation accuracy:', means[:1 +  FLAGS.test_num_updates])
            print('Mean validation 95_range:', ci95[:1 +  FLAGS.test_num_updates])
            print('----------------------------------------------', )

            print('Mean validation APCER:', means[   (FLAGS.test_num_updates+1):  2 *  (FLAGS.test_num_updates+1)])
            print('Mean validation APCER_95:', ci95[   (FLAGS.test_num_updates+1):  2 *  (FLAGS.test_num_updates+1)])
            print('Mean validation NPCER:', means[  2 *  (FLAGS.test_num_updates+1):  3 *  (FLAGS.test_num_updates+1)])
            print('Mean validation NPCER_95:', ci95[  2 *  (FLAGS.test_num_updates+1):  3 *  (FLAGS.test_num_updates+1)])
            print('Mean validation ACER:', means[  3 *  (FLAGS.test_num_updates+1): ])
            print('Mean validation ACER_95:', ci95[  3 *  (FLAGS.test_num_updates+1): ])

            if min_APCER > min(means[   (FLAGS.test_num_updates+1):  2 *  (FLAGS.test_num_updates+1)]):
                min_APCER = min(means[   (FLAGS.test_num_updates+1):  2 *  (FLAGS.test_num_updates+1)])
                index = np.where(means[   (FLAGS.test_num_updates+1):  2 *  (FLAGS.test_num_updates+1)] == min_APCER)
                APCER_95 = ci95[   (FLAGS.test_num_updates+1):  2 *  (FLAGS.test_num_updates+1)][index][0]
                min_APCER_itr = itr
            print('Min validation APCER is  :', min_APCER, '  ;', '95% range is:', APCER_95, '  ;', 'iteration is:',
                  min_APCER_itr)

            if min_NPCER > min(means[  2 *  (FLAGS.test_num_updates+1):  3 *  (FLAGS.test_num_updates+1)]):
                min_NPCER = min(means[  2 *  (FLAGS.test_num_updates+1):  3 *  (FLAGS.test_num_updates+1)])
                index = np.where(means[  2 *  (FLAGS.test_num_updates+1):  3 *  (FLAGS.test_num_updates+1)] == min_NPCER)
                NPCER_95 = ci95[  2 *  (FLAGS.test_num_updates+1):  3 *  (FLAGS.test_num_updates+1)][index][0]
                min_NPCER_itr = itr
            print('Min validation FRR is  :', min_NPCER, '  ;', '95% range is:', NPCER_95, '  ;', 'iteration is:',
                  min_NPCER_itr)

            if min_ACER > min(means[  3 *  (FLAGS.test_num_updates+1):  ]):
                min_ACER = min(means[  3 *  (FLAGS.test_num_updates+1):  ])
                index = np.where(means[  3 *  (FLAGS.test_num_updates+1):  ] == min_ACER)
                ACER_95 = ci95[  3 *  (FLAGS.test_num_updates+1):  4 *  (FLAGS.test_num_updates+1)][index][0]
                min_ACER_itr = itr
            print('Min validation HTER is :', min_ACER, '  ;', '95% range is:', ACER_95, '  ;', 'iteration is:',
                  min_ACER_itr)

            if min_ACER_pre > means[3*(FLAGS.test_num_updates+1)+1]:
                min_ACER_pre = means[3*(FLAGS.test_num_updates+1)+1]
                ACER_95_pre = ci95[3*(FLAGS.test_num_updates+1)+1]
                min_ACER_pre_itr = itr
            print('Min validation ACER is :', min_ACER_pre, '  ;', '95% range is:', ACER_95_pre, '  ;', 'iteration is:',
                  min_ACER_pre_itr)

            print('----------------------------------------', )

            saver.save(sess, FLAGS.logdir + '/' + hyper_setting +  '/model' + str(itr))
    saver.save(sess, FLAGS.logdir + '/' + hyper_setting +  '/model' + str(itr))


def test(model, sess, task_generator):
    metaval_accuracies = []
    for test_itr in range(NUM_TEST_POINTS):
        feed_dict_data_test = {}
        feed_dict_test = {model.meta_lr: 0}
        if test_itr == 0:
            metaval_train_files, metaval_test_files = task_generator.get_data_n_tasks(
                FLAGS.num_gpus * FLAGS.meta_batch_size, train=False)
            for task_id in range(FLAGS.meta_batch_size * FLAGS.num_gpus):
                im_file = metaval_train_files[task_id]
                im_file_test = metaval_test_files[task_id]
                im_file.extend(im_file_test)
                feed_dict_data_test[task_generator.image_lists[task_id]] = im_file
            sess.run(task_generator.iterators, feed_dict=feed_dict_data_test)
            [metaval_ims, metaval_depthes] = sess.run([task_generator.out_faces, task_generator.out_depthes])

            metaval_train_ims = metaval_ims[:, :FLAGS.num_classes * FLAGS.num_support, :]
            metaval_test_ims = metaval_ims[:, FLAGS.num_classes * FLAGS.num_support:, :]
            metaval_train_lbls = metaval_depthes[:, :FLAGS.num_classes * FLAGS.num_support, :]
            metaval_test_lbls = metaval_depthes[:, FLAGS.num_classes * FLAGS.num_support:, :]

        feed_dict_test[model.inputa] = metaval_train_ims
        feed_dict_test[model.inputb] = metaval_test_ims
        feed_dict_test[model.labela] = metaval_train_lbls
        feed_dict_test[model.labelb] = metaval_test_lbls

        metaval_train_files, metaval_test_files = task_generator.get_data_n_tasks(
            FLAGS.num_gpus * FLAGS.meta_batch_size, train=False)
        for task_id in range(FLAGS.meta_batch_size * FLAGS.num_gpus):
            im_file = metaval_train_files[task_id]
            im_file_test = metaval_test_files[task_id]
            im_file.extend(im_file_test)
            feed_dict_data_test[task_generator.image_lists[task_id]] = im_file
        sess.run(task_generator.iterators, feed_dict=feed_dict_data_test)

        input_tensors = [[model.metaval_total_loss1] + model.metaval_total_losses2 +
                         model.metaval_APCER + model.metaval_NPCER + model.metaval_ACER,
                         task_generator.out_faces, task_generator.out_depthes]

        result = sess.run(input_tensors, feed_dict_test)
        metaval_accuracies.append(result[0])
        metaval_ims = result[-2]
        metaval_depthes = result[-1]
        metaval_train_ims = metaval_ims[:, :FLAGS.num_classes * FLAGS.num_support, :]
        metaval_test_ims = metaval_ims[:, FLAGS.num_classes * FLAGS.num_support:, :]
        metaval_train_lbls = metaval_depthes[:, :FLAGS.num_classes * FLAGS.num_support, :]
        metaval_test_lbls = metaval_depthes[:, FLAGS.num_classes * FLAGS.num_support:, :]

    metaval_accuracies = np.array(metaval_accuracies)
    means = np.mean(metaval_accuracies, 0)
    stds = np.std(metaval_accuracies, 0)
    ci95 = 1.96 * stds / np.sqrt(NUM_TEST_POINTS)
    print('----------------------------------------------', )
    print('Mean validation accuracy:', means[:1 + FLAGS.test_num_updates])
    print('Mean validation 95_range:', ci95[:1 + FLAGS.test_num_updates])
    print('----------------------------------------------', )

    print('Mean validation APCER:', means[(FLAGS.test_num_updates + 1):  2 * (FLAGS.test_num_updates + 1)])
    print('Mean validation APCER_95:', ci95[(FLAGS.test_num_updates + 1):  2 * (FLAGS.test_num_updates + 1)])
    print('Mean validation NPCER:', means[2 * (FLAGS.test_num_updates + 1):  3 * (FLAGS.test_num_updates + 1)])
    print('Mean validation NPCER_95:', ci95[2 * (FLAGS.test_num_updates + 1):  3 * (FLAGS.test_num_updates + 1)])
    print('Mean validation ACER:', means[3 * (FLAGS.test_num_updates + 1):  ])
    print('Mean validation ACER_95:', ci95[3 * (FLAGS.test_num_updates + 1):  ])
    print('Mean validation ACER_pre:',  means[3 * (FLAGS.test_num_updates + 1) + 1])
    acer = min(means[3 * (FLAGS.test_num_updates + 1) + 1:  ])

    return acer



def main():
    FLAGS.logdir = FLAGS.logdir + str(FLAGS.num_support) + '/'

    print('preparing data')
    task_generator = Task_dataset()   #define the task generator

    print('initializing the model')
    model = AIM_FAS()
    if FLAGS.train:
        model.construct_model(num_updates=FLAGS.num_updates, train=True)
    model.construct_model(num_updates=FLAGS.test_num_updates, train=False)

    saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=0)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)

    hyper_setting = str(FLAGS.meta_batch_size*FLAGS.num_gpus) + '.lr' + str(FLAGS.meta_lr) + '.ilr' + str(FLAGS.update_lr)
    hyper_setting += '.ns_' + str(FLAGS.num_updates) + '.nts' + str(FLAGS.num_train_tasks)
    hyper_setting += '.ubs' + str(FLAGS.num_support) + '_' + str(FLAGS.num_query_t)+ '_' + str(FLAGS.num_query_v)
    hyper_setting += '.nfs' + str(FLAGS.base_num_filters)
    hyper_setting += '.l1_' + str(FLAGS.l1_alpha) +'.l2_' + str(FLAGS.l2_alpha)
    hyper_setting += '.lb' + str(FLAGS.loss) + '.inl' + str(FLAGS.inner_losses)
    hyper_setting += '.sht' + str(FLAGS.shot_list)

    if FLAGS.restore:
        hyper_setting += '.R'

    if FLAGS.lr_decay_itr > 0:
        hyper_setting += '.decay' + str(FLAGS.lr_decay_itr/1000)

    resume_itr = 0
    tf.global_variables_initializer().run()

    if FLAGS.resume:
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + hyper_setting)
        if FLAGS.test_iter > 0:
            model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.test_iter)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1 + 5:])
            print("Restoring model from " + model_file)
            saver.restore(sess, model_file)

    elif FLAGS.train and FLAGS.restore:
        checkpoint_path = FLAGS.pretrain_model
        reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        var_list = []
        for key in var_to_shape_map:
            var_list.append(key)

        for var in variables:
            name = var.name[:-2]
            new_name = name

            if 'Adam' in new_name or 'moving' in new_name or 'pow' in new_name or 'bn4' in new_name or 'alpha' in new_name or 'decay' in new_name:
                pass
            elif new_name in var_list:
                print('loading weights of the var:', new_name)
                var.load(reader.get_tensor(new_name))
            else:
                print('var_name', new_name)
                raise ValueError('var name not recognized, please check the name:', new_name)
    else:
        pass


    if FLAGS.train:
        train(model, saver, sess, hyper_setting, task_generator, resume_itr)
    else:
        model_file = FLAGS.logdir + hyper_setting + '/model' + str(FLAGS.test_iter)
        saver.restore(sess, model_file)
        print(str(datetime.datetime.now())[:-7], "testing model: " + model_file)
        acer = test(model, sess, task_generator)
        print('----------test acer:', acer, '------------------------')

    tf.reset_default_graph()

if __name__ == "__main__":
    main()
