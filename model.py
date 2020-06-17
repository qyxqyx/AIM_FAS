from __future__ import print_function
import numpy as np
import tensorflow as tf
import copy
from sklearn.metrics import roc_curve
from scipy import interp
from tensorflow.python.platform import flags
from utils import CDL, L2_loss
from networks import ZZNet

FLAGS = flags.FLAGS

def average_gradients(tower_grads):
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    grads = []
    for g, _ in grad_and_vars:
      expanded_g = tf.expand_dims(g, 0)
      grads.append(expanded_g)
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def get_err_threhold(fpr, tpr, threshold):
    differ_tpr_fpr_1=tpr+fpr-1.0
    right_index = np.argmin(np.abs(differ_tpr_fpr_1))
    best_th = threshold[right_index]
    err = fpr[right_index]

    return err, best_th


def performances(test_scores, test_labels):
    # print('label',test_labels)
    # print('score',test_scores)

    test_labels_bk = copy.deepcopy(test_labels)
    test_scores_bk = copy.deepcopy(test_scores)
    test_labels_bk[test_labels_bk < 0] = 0

    fpr_test, tpr_test, threshold_test = roc_curve(test_labels_bk, test_scores_bk, pos_label=1)
    err, best_th = get_err_threhold(fpr_test, tpr_test, threshold_test)
    precision_th1 = 0.005
    RECALL1 = interp(precision_th1, fpr_test, tpr_test)

    precision_th2 = 0.01
    RECALL2 = interp(precision_th2, fpr_test, tpr_test)

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(test_labels.shape[0]):
        if test_labels[i] == 0:
            if test_scores[i] < best_th:
                TN += 1
            else:
                FP += 1
        else:
            if test_scores[i] >= best_th:
                TP += 1
            else:
                FN += 1

    APCER = FP / (TN + FP + 0.000001)  ### Attack Presentation Classification Error Rate
    NPCER = FN / (FN + TP + 0.000001)  ### Normal Presentation Classification Error Rate
    ACER = (APCER + NPCER) / 2  ### Average Classification Error Rate

    return np.float32(APCER), np.float32(NPCER), np.float32(ACER)



class AIM_FAS(object):
    def __init__(self):
        self.meta_lr = tf.placeholder_with_default(FLAGS.meta_lr, ())
        self.net = ZZNet()
        self.forward = self.net.forward
        self.construct_weights = self.net.construct_weights

        if FLAGS.loss == 'L2':
            self.loss_func = L2_loss
        else:
            self.loss_func = CDL

        shape = [FLAGS.num_gpus * FLAGS.meta_batch_size, None, 256, 256, 3]
        self.inputa = tf.placeholder(tf.float32, shape=shape)
        shape = [FLAGS.num_gpus * FLAGS.meta_batch_size, None, 256, 256, 3]
        self.inputb = tf.placeholder(tf.float32, shape=shape)
        shape = [FLAGS.num_gpus * FLAGS.meta_batch_size, None, 32,32,1]
        self.labela = tf.placeholder(tf.float32, shape=shape)
        shape = [FLAGS.num_gpus * FLAGS.meta_batch_size, None, 32,32,1]
        self.labelb = tf.placeholder(tf.float32, shape=shape)

        alpha_initializer = tf.initializers.random_uniform(minval=FLAGS.update_lr * 0.8, maxval=FLAGS.update_lr * 1.25)
        decay_initializer = tf.initializers.ones()
        self.alpha = tf.get_variable('alpha', shape=[1, ], dtype=tf.float32, initializer=alpha_initializer)
        self.decay = tf.get_variable('decay', shape=[1, ], dtype=tf.float32, initializer=decay_initializer)


    def construct_model(self, num_updates=1, train=True):
        # a: training data for inner gradient, b: test data for meta gradient
        self.net.train_flag = train

        optimizer = tf.train.AdamOptimizer(self.meta_lr)

        total_loss1_gpus = []
        total_losses2_gpus = []
        total_APCER_gpus = []
        total_NPCER_gpus = []
        total_ACER_gpus = []
        tower_grads = []

        inputas = tf.split(self.inputa, num_or_size_splits=FLAGS.num_gpus, axis=0)
        inputbs = tf.split(self.inputb, num_or_size_splits=FLAGS.num_gpus, axis=0)
        labelas = tf.split(self.labela, num_or_size_splits=FLAGS.num_gpus, axis=0)
        labelbs = tf.split(self.labelb, num_or_size_splits=FLAGS.num_gpus, axis=0)

        self.weights_1, self.weights_2, self.weights_3, self.weights_4 = self.construct_weights()
        self.weights = {}
        self.weights.update(self.weights_1)
        self.weights.update(self.weights_2)
        self.weights.update(self.weights_3)
        self.weights.update(self.weights_4)
        weights = self.weights

        def meta_learner(inp, reuse=True):
            '''
            :param inp:  the support and query data for the current task
            :param reuse: reuse the network weights?
            :return: the meta-learner's output, loss, performances on the current task.
            '''
            inputa, inputb, labela, labelb = inp

            task_outputbs, task_lossesb = [], []

            task_accuraciesb2_APCER = []
            task_accuraciesb2_NPCER = []
            task_accuraciesb2_ACER = []

            task_outputa = self.forward(inputa, weights, reuse=reuse)  # only reuse on the first iter
            task_lossa = self.loss_func(task_outputa, labela)

            grads = tf.gradients(task_lossa, list(weights.values()))
            grads = [tf.stop_gradient(grad) for grad in grads]
            gradients = dict(zip(weights.keys(), grads))

            fast_weights = dict(
                zip(weights.keys(), [weights[key] - self.alpha * gradients[key] for key in weights.keys()]))
            task_outputbs.append(self.forward(inputb, weights, reuse=True))
            output = self.forward(inputb, fast_weights, reuse=True)
            task_outputbs.append(output)
            task_lossesb.append(self.loss_func(output, labelb))

            for j in range(num_updates - 1):
                loss = self.loss_func(self.forward(inputa, fast_weights, reuse=True), labela)
                grads = tf.gradients(loss, list(fast_weights.values()))
                grads = [tf.stop_gradient(grad) for grad in grads]
                gradients = dict(zip(fast_weights.keys(), grads))

                fast_weights = dict(zip(fast_weights.keys(),
                                        [fast_weights[key] - self.alpha * tf.pow(self.decay, j + 1) * gradients[key] for
                                         key in fast_weights.keys()]))

                output = self.forward(inputb, fast_weights, reuse=True)
                task_outputbs.append(output)
                task_lossesb.append(self.loss_func(output, labelb))

            task_output = [task_lossa, task_lossesb]

            for j in range(num_updates + 1):
                predict = tf.reduce_mean(task_outputbs[j], axis=[1, 2, 3])

                true_label = tf.reduce_mean(labelb, axis=[1, 2, 3])
                true_label = tf.greater(true_label, 0.05)
                true_label = tf.cast(true_label, dtype=tf.uint8)
                APCER, NPCER, ACER = tf.py_func(performances,
                                                inp=[predict, true_label],
                                                Tout=[tf.float32, tf.float32, tf.float32])
                task_accuraciesb2_APCER.append(APCER)
                task_accuraciesb2_NPCER.append(NPCER)
                task_accuraciesb2_ACER.append(ACER)

            task_output.extend([task_accuraciesb2_APCER, task_accuraciesb2_NPCER, task_accuraciesb2_ACER])

            return task_output

        _ = meta_learner((inputas[0][0], inputbs[0][0], labelas[0][0],
                               labelbs[0][0]), False)

        out_dtype = [tf.float32, [tf.float32] * num_updates]

        out_dtype.extend(
            [[tf.float32] * (num_updates + 1), [tf.float32] * (num_updates + 1), [tf.float32] * (num_updates + 1)], )


        for gpu_id in range(FLAGS.num_gpus):
            with tf.device('/gpu:%d' % gpu_id):
                with tf.variable_scope('', reuse=tf.AUTO_REUSE) as training_scope:
                    result = tf.map_fn(meta_learner, elems=(inputas[gpu_id], inputbs[gpu_id], labelas[gpu_id], labelbs[gpu_id]), dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
                    lossesa, lossesb, APCER, NPCER, ACER = result

                total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
                total_loss1_gpus.append(total_loss1)
                total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in
                                 range(num_updates)]
                total_losses2_gpus.append(total_losses2)

                total_APCER = [tf.reduce_sum(APCER[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range((num_updates+1))]
                total_NPCER = [tf.reduce_sum(NPCER[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range((num_updates+1))]
                total_ACER = [tf.reduce_sum(ACER[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range((num_updates+1))]
                total_APCER_gpus.append(total_APCER)
                total_NPCER_gpus.append(total_NPCER)
                total_ACER_gpus.append(total_ACER)

                tf.get_variable_scope().reuse_variables()

                if train:
                    weight_l_loss = 0
                    if FLAGS.l2_alpha > 0:
                        for key, array in self.weights.items():
                            weight_l_loss += tf.reduce_sum(tf.square(array)) * FLAGS.l2_alpha
                    if FLAGS.l1_alpha > 0:
                        for key, array in self.weights.items():
                            weight_l_loss += tf.reduce_sum(tf.abs(array)) * FLAGS.l1_alpha

                    weight_list = list(self.weights.values())
                    weight_list.append(self.alpha)
                    weight_list.append(self.decay)

                    if ',' in FLAGS.inner_losses:
                        inner_loss_indexes = FLAGS.inner_losses.split(',')
                    else:
                        inner_loss_indexes = [FLAGS.inner_losses]

                    inner_loss = 0
                    for index in inner_loss_indexes:
                        loss = total_losses2[int(index)]
                        inner_loss += loss

                    gvs1 = optimizer.compute_gradients(inner_loss + weight_l_loss, weight_list)
                    gvs1 = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in gvs1]
                    tower_grads.append(gvs1)

        mean_loss1 = tf.stack(axis=0, values=total_loss1_gpus)
        mean_losses2 = [tf.stack(axis=0, values=list(losses)) for losses in list(zip(*total_losses2_gpus))]
        mean_APCER = [tf.stack(axis=0, values=list(APCER)) for APCER in list(zip(*total_APCER_gpus))]
        mean_NPCER = [tf.stack(axis=0, values=list(NPCER)) for NPCER in list(zip(*total_NPCER_gpus))]
        mean_ACER = [tf.stack(axis=0, values=list(ACER)) for ACER in list(zip(*total_ACER_gpus))]

        ## Performance & Optimization
        if train:
            self.total_loss1 = tf.reduce_mean(mean_loss1, 0)
            self.total_losses2 = [tf.reduce_mean(losses, 0) for losses in mean_losses2]

            self.APCER = [tf.reduce_mean(accs, 0) for accs in mean_APCER]
            self.NPCER = [tf.reduce_mean(accs, 0) for accs in mean_NPCER]
            self.ACER = [tf.reduce_mean(accs, 0) for accs in mean_ACER]

            mean_grads = average_gradients(tower_grads)

            with tf.variable_scope('', reuse=tf.AUTO_REUSE):
                self.metatrain_op = optimizer.apply_gradients(mean_grads)

        else:
            self.metaval_total_loss1 = tf.reduce_mean(mean_loss1, 0)
            self.metaval_total_losses2 = [tf.reduce_mean(losses, 0) for losses in mean_losses2]

            self.metaval_APCER = [tf.reduce_mean(accs, 0) for accs in mean_APCER]
            self.metaval_NPCER = [tf.reduce_mean(accs, 0) for accs in mean_NPCER]
            self.metaval_ACER = [tf.reduce_mean(accs, 0) for accs in mean_ACER]


