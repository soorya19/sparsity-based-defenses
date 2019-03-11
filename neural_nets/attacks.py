import numpy as np
from numpy.random import RandomState
from tqdm import tqdm

import tensorflow as tf
from cleverhans.model import Model

from sp_func_nn import sp_project, sp_project_binary

class LocLinearAttack:
    """
    Locally linear attack. 

    Given the true label l, it finds the locally linear attacks targeting each 
    of the other labels t =/= l by calculating epsilon*sgn(weq_t - weq_l). 
    It then picks the attack with the highest output distortion.
    """
    def __init__(self, model, sess):
        """
        Inputs:
            model   - An instance of the cleverhans.model.Model class.
            sess    - The tf.Session to run graphs in.
        """
        
        if not isinstance(model, Model):
            raise TypeError('The model argument should be an instance of the cleverhans.model.Model class.')

        self.model = model
        self.sess = sess

    def run(self, x, y, batch_size, eps = 0.2,
                                    clip_min = 0.,
                                    clip_max = 1.,
                                    progress = True):
        """
        Inputs:
            x           - Images to be perturbed. Must be a Numpy array of shape 
                          [num_images, num_rows, num_cols, num_channels]
            y           - True labels, one-hot encoded. Must be a Numpy array of shape
                          [num_images, num_classes]
            eps         - L-inf budget for the attack.
            clip_min    - Minimum input component value.
            clip_max    - Maximum input component value.
            progress    - Whether to display progress bars

        Output:
            x_adv       - Perturbed image. Numpy array of same shape as x.
        """

        num_images, num_rows, num_cols, num_channels = x.shape
        num_classes = y.shape[1]

        # Create graph
        x_inp = tf.placeholder(tf.float32, shape=(None, num_rows, num_cols, num_channels))
        logits = self.model.get_logits(x_inp)

        x_adv = x.copy()
        x_adv_t = np.zeros([num_classes, num_images, num_rows, num_cols, num_channels])
        weq_t = np.zeros([num_classes, num_images, num_rows, num_cols, num_channels])
        logits_t = np.zeros([num_images, num_classes])
        labels = np.argmax(y, axis=1)

        seq_batches = range(np.int(num_images/batch_size))
        seq_images = range(num_images)
        seq_targets = range(num_classes)
        seq_images_2 = range(num_images)
        seq_targets_2 = range(num_classes)
        if progress is True:
            seq_targets = tqdm(seq_targets)
            seq_targets_2 = tqdm(seq_targets_2)

        t_ = tf.placeholder(tf.int32, shape=())
        grad_t, = tf.gradients(logits[:, t_], x_inp)

        for t in seq_targets:
            for i in seq_batches:
                batch = slice(i*batch_size, (i+1)*batch_size)
                weq_t[t, batch] = self.sess.run(grad_t, feed_dict={x_inp: x_adv[batch], t_: t})
        
        for i in seq_images:
            l = labels[i]
            for t in range(num_classes):
                x_adv_t[t, i] = x[i] + eps * np.sign(weq_t[t, i] - weq_t[l, i])
        x_adv_t = np.clip(x_adv_t, clip_min, clip_max)

        for t in seq_targets_2:
            logits_all = self.sess.run(logits, feed_dict={x_inp: x_adv_t[t]})
            logits_t[:, t] = logits_all[:, t]
            for i in seq_images:
                l = labels[i]
                logits_t[i, t] -= logits_all[i, l]
                logits_t[i, l] = -np.inf

        targets = logits_t.argmax(axis=1)
        for i in seq_images_2:
            x_adv[i] = x_adv_t[targets[i], i]

        return x_adv

    def run_proj(self, x, y, batch_size, niter_proj, 
                                         def_params,
                                         eps = 0.2,
                                         clip_min = 0.,
                                         clip_max = 1.,
                                         progress = True):
        """
        LocLinearAttack with iterated projections for gradient of front end

        Inputs:
            x           - Images to be perturbed. Must be a Numpy array of shape 
                          [num_images, num_rows, num_cols, num_channels]
            y           - True labels, one-hot encoded. Must be a Numpy array of shape
                          [num_images, num_classes]
            niter_proj  - No of iterations for projection refinement
            def_params  - Defense parameters to pass to sp_project
            eps         - L-inf budget for the attack.
            clip_min    - Minimum input component value.
            clip_max    - Maximum input component value.
            progress    - Whether to display progress bars

        Output:
            x_adv       - Perturbed image. Numpy array of same shape as x.
        """

        num_images, num_rows, num_cols, num_channels = x.shape
        num_classes = y.shape[1]

        # Create graph
        x_inp = tf.placeholder(tf.float32, shape=(None, num_rows, num_cols, num_channels))
        logits = self.model.get_logits(x_inp)

        x_adv = x.copy()
        x_adv_t = np.tile(x[None, ...], [num_classes, 1, 1, 1, 1])
        # x_adv_t = np.zeros([num_classes, num_images, num_rows, num_cols, num_channels])
        weq_t = np.zeros([num_classes, num_images, num_rows, num_cols, num_channels])
        logits_t = np.zeros([num_images, num_classes])
        labels = np.argmax(y, axis=1)

        seq_batches = range(np.int(num_images/batch_size))
        seq_images = range(num_images)
        seq_targets = range(num_classes)
        seq_targets_2 = range(num_classes)
        if progress is True:
            seq_targets = tqdm(seq_targets)

        t_ = tf.placeholder(tf.int32, shape=())
        grad_t, = tf.gradients(logits[:, t_], x_inp)

        for t in seq_targets:
            for i in seq_batches:
                batch = slice(i*batch_size, (i+1)*batch_size)
                weq_t[t, batch] = self.sess.run(grad_t, feed_dict={x_inp: x_adv[batch], t_: t})
        
        for i in seq_images:
            l = labels[i]
            for t in range(num_classes):
                x_adv_t[t, i] = x[i] + eps * np.sign(weq_t[t, i] - weq_t[l, i])
        x_adv_t = np.clip(x_adv_t, clip_min, clip_max)

        weq_t_proj = np.zeros([num_classes, num_images, num_rows, num_cols, num_channels])
        for j in tqdm(range(niter_proj)):
            for t in range(num_classes):
                weq_t_proj[t] = sp_project(x_adv_t[t], weq_t[t], **def_params)
            for i in seq_images:
                l = labels[i]
                for t in range(num_classes):
                    x_adv_t[t, i] = x[i] + eps * np.sign(weq_t_proj[t, i] - weq_t_proj[l, i])
            x_adv_t = np.clip(x_adv_t, clip_min, clip_max)

        if progress is True:
            seq_targets_2 = tqdm(seq_targets_2)
        for t in seq_targets_2:
            logits_all = self.sess.run(logits, feed_dict={x_inp: x_adv_t[t]})
            logits_t[:, t] = logits_all[:, t]
            for i in seq_images:
                l = labels[i]
                logits_t[i, t] -= logits_all[i, l]
                logits_t[i, l] = -np.inf

        targets = logits_t.argmax(axis=1)
        for i in seq_images:
            x_adv[i] = x_adv_t[targets[i], i]

        return x_adv

    def run_binary(self, x, y, batch_size, eps = 0.2,
                                           clip_min = 0.,
                                           clip_max = 1.,
                                           progress = True):
        """
        LocLinearAttack for binary (sigmoid) classification with fully connected layers

        Inputs:
            x           - Images to be perturbed. Must be a Numpy array of shape 
                          [num_images, dim]
            y           - True labels, either 0 or 1. Must be a Numpy array of shape
                          [num_images]
            eps         - L-inf budget for the attack.
            clip_min    - Minimum input component value.
            clip_max    - Maximum input component value.
            progress    - Whether to display progress bars

        Output:
            x_adv       - Perturbed image. Numpy array of same shape as x.
        """

        num_images, dim = x.shape

        # Create graph
        x_inp = tf.placeholder(tf.float32, shape=(None, dim))
        logits = self.model.get_logits(x_inp)
        grad, = tf.gradients(logits, x_inp)

        seq_batches = range(np.int(num_images/batch_size))
        if progress is True:
            seq_batches = tqdm(seq_batches)

        weq = np.zeros([num_images, dim])
        for i in seq_batches:
            batch = slice(i*batch_size, (i+1)*batch_size)
            weq[batch] = self.sess.run(grad, feed_dict={x_inp: x[batch]})

        # weq = self.sess.run(grad, feed_dict={x_inp: x})

        sgn_y = 1.0 - 2.0*y
        # sgn_y = - np.sign(2.0*y - 1.0)
        x_adv = x + eps * np.sign(weq) * sgn_y
        x_adv = np.clip(x_adv, clip_min, clip_max)

        return x_adv

    def run_binary_proj(self, x, y, batch_size, niter_proj, 
                                                def_params,
                                                eps = 0.2,
                                                clip_min = 0.,
                                                clip_max = 1.,
                                                progress = True):
        """
        LocLinearAttack for binary (sigmoid) classification with fully connected layers, 
        using iterated projections for gradient of front end

        Inputs:
            x           - Images to be perturbed. Must be a Numpy array of shape 
                          [num_images, dim]
            y           - True labels, either 0 or 1. Must be a Numpy array of shape
                          [num_images]
            niter_proj  - No of iterations for projection refinement
            def_params  - Defense parameters to pass to sp_project
            eps         - L-inf budget for the attack.
            clip_min    - Minimum input component value.
            clip_max    - Maximum input component value.
            progress    - Whether to display progress bars

        Output:
            x_adv       - Perturbed image. Numpy array of same shape as x.
        """

        num_images, dim = x.shape

        # Create graph
        x_inp = tf.placeholder(tf.float32, shape=(None, dim))
        logits = self.model.get_logits(x_inp)
        grad, = tf.gradients(logits, x_inp)

        seq_batches = range(np.int(num_images/batch_size))
        if progress is True:
            seq_batches = tqdm(seq_batches)

        weq = np.zeros([num_images, dim])
        for i in seq_batches:
            batch = slice(i*batch_size, (i+1)*batch_size)
            weq[batch] = self.sess.run(grad, feed_dict={x_inp: x[batch]})

        # weq = self.sess.run(grad, feed_dict={x_inp: x})

        sgn_y = 1.0 - 2.0*y

        x_adv = x + eps * np.sign(weq) * sgn_y
        x_adv = np.clip(x_adv, clip_min, clip_max)

        for j in tqdm(range(niter_proj)):
            weq_proj = sp_project_binary(x_adv, weq, **def_params)
            x_adv = x + eps * np.sign(weq_proj) * sgn_y
            x_adv = np.clip(x_adv, clip_min, clip_max)

        return x_adv


class IterLocLinearAttack:
    """
    Iterative version of locally linear attack. 

    Given the true label l, it finds the iterative locally linear attacks targeting 
    each of the other labels t =/= l by calculating delta*sgn(weq_t - weq_l) at each
    iteration. It then picks the attack with the highest output distortion.
    """
    def __init__(self, model, sess):
        """
        Inputs:
            model   - An instance of the cleverhans.model.Model class.
            sess    - The tf.Session to run graphs in.
        """
        
        if not isinstance(model, Model):
            raise TypeError('The model argument should be an instance of the cleverhans.model.Model class.')

        self.model = model
        self.sess = sess

    def run(self, x, y, batch_size, eps = 0.2,
                                    delta = 0.01,
                                    niter = 1000,
                                    clip_min = 0.,
                                    clip_max = 1.,
                                    progress = True):
        """
        Inputs:
            x           - Images to be perturbed. Must be a Numpy array of shape 
                          [num_images, num_rows, num_cols, num_channels]
            y           - True labels, one-hot encoded. Must be a Numpy array of shape
                          [num_images, num_classes]
            eps         - Overall L-inf budget for the attack.
            delta       - L-inf budget for each attack iteration.
            niter       - Number of attack iterations.
            clip_min    - Minimum input component value.
            clip_max    - Maximum input component value.
            progress    - Whether to display progress bars

        Output:
            x_adv       - Perturbed image. Numpy array of same shape as x.
        """

        num_images, num_rows, num_cols, num_channels = x.shape
        num_classes = y.shape[1]

        # Create graph
        x_inp = tf.placeholder(tf.float32, shape=(None, num_rows, num_cols, num_channels))
        logits = self.model.get_logits(x_inp)

        x_adv = x.copy()
        seq_niter = range(niter)

        x_adv = x.copy()
        x_tiled = np.tile(x[None, ...], [num_classes, 1, 1, 1, 1])
        x_adv_t = x_tiled.copy()
        weq_t = np.zeros([num_classes, num_images, num_rows, num_cols, num_channels])
        logits_all = np.zeros([num_images, num_classes])
        logits_t = np.zeros([num_images, num_classes])
        labels = np.argmax(y, axis=1)

        seq_niter = range(niter)
        seq_batches = range(np.int(num_images/batch_size))
        seq_images = range(num_images)
        seq_targets = range(num_classes)
        seq_targets_2 = range(num_classes)

        t_ = tf.placeholder(tf.int32, shape=())
        grad_t, = tf.gradients(logits[:, t_], x_inp)

        if progress is True:
            seq_niter = tqdm(seq_niter)

        for n in seq_niter:
            if progress is True:
                seq_targets = tqdm(range(num_classes))

            for t in seq_targets:
                # grad_t, = tf.gradients(logits[:, t], x_inp)
                for i in seq_batches:
                    batch = slice(i*batch_size, (i+1)*batch_size)
                    weq_t[t, batch] = self.sess.run(grad_t, feed_dict={x_inp: x_adv_t[t, batch], t_: t})

            for i in seq_images:
                l = labels[i]
                for t in range(num_classes):
                    x_adv_t[t, i] += delta * np.sign(weq_t[t, i] - weq_t[l, i])

            for t in range(num_classes):
                for i in seq_images:
                    l = labels[i]
                    x_adv_t[t, i] += delta * np.sign(weq_t[t, i] - weq_t[l, i])

            x_adv_t = np.clip(x_adv_t, x_tiled - eps, x_tiled + eps)
            x_adv_t = np.clip(x_adv_t, clip_min, clip_max)

        if progress is True:
            seq_targets_2 = tqdm(range(num_classes))

        for t in seq_targets_2:
            for i in seq_batches:
                batch = slice(i*batch_size, (i+1)*batch_size)
                logits_all[batch] = self.sess.run(logits, feed_dict={x_inp: x_adv_t[t, batch]})
            logits_t[:, t] = logits_all[:, t]
            for i in seq_images:
                l = labels[i]
                logits_t[i, t] -= logits_all[i, l]
                logits_t[i, l] = -np.inf

        targets = logits_t.argmax(axis=1)
        for i in seq_images:
            x_adv[i] = x_adv_t[targets[i], i]

        return x_adv

    def run_proj(self, x, y, batch_size, niter_proj, 
                                         def_params,
                                         eps = 0.2,
                                         delta = 0.01,
                                         niter = 1000,
                                         clip_min = 0.,
                                         clip_max = 1.,
                                         progress = True):
        """
        IterLocLinearAttack with iterated projections for gradient of front end

        Inputs:
            x           - Images to be perturbed. Must be a Numpy array of shape 
                          [num_images, num_rows, num_cols, num_channels]
            y           - True labels, one-hot encoded. Must be a Numpy array of shape
                          [num_images, num_classes]
            niter_proj  - No of iterations for projection refinement
            def_params  - Defense parameters to pass to sp_project
            eps         - L-inf budget for the attack.
            clip_min    - Minimum input component value.
            clip_max    - Maximum input component value.
            progress    - Whether to display progress bars

        Output:
            x_adv       - Perturbed image. Numpy array of same shape as x.
        """

        num_images, num_rows, num_cols, num_channels = x.shape
        num_classes = y.shape[1]

        # Create graph
        x_inp = tf.placeholder(tf.float32, shape=(None, num_rows, num_cols, num_channels))
        logits = self.model.get_logits(x_inp)

        x_adv = x.copy()
        seq_niter = range(niter)

        x_adv = x.copy()
        x_tiled = np.tile(x[None, ...], [num_classes, 1, 1, 1, 1])
        x_adv_t = x_tiled.copy()
        weq_t = np.zeros([num_classes, num_images, num_rows, num_cols, num_channels])
        logits_all = np.zeros([num_images, num_classes])
        logits_t = np.zeros([num_images, num_classes])
        labels = np.argmax(y, axis=1)

        seq_niter = range(niter)
        seq_batches = range(np.int(num_images/batch_size))
        seq_images = range(num_images)
        seq_targets = range(num_classes)
        seq_images_2 = range(num_images)
        seq_targets_2 = range(num_classes)

        t_ = tf.placeholder(tf.int32, shape=())
        grad_t, = tf.gradients(logits[:, t_], x_inp)

        if progress is True:
            seq_niter = tqdm(seq_niter)

        weq_t_proj = np.zeros([num_classes, num_images, num_rows, num_cols, num_channels])
        for n in seq_niter:
            x_adv_t_current = x_adv_t.copy()

            for t in tqdm(range(num_classes)):
                for i in seq_batches:
                    batch = slice(i*batch_size, (i+1)*batch_size)
                    weq_t[t, batch] = self.sess.run(grad_t, feed_dict={x_inp: x_adv_t[t, batch], t_: t})

            for i in seq_images:
                l = labels[i]
                for t in range(num_classes):
                    x_adv_t[t, i] += delta * np.sign(weq_t[t, i] - weq_t[l, i])
            x_adv_t = np.clip(x_adv_t, x_tiled - eps, x_tiled + eps)
            x_adv_t = np.clip(x_adv_t, clip_min, clip_max)

            for j in tqdm(range(niter_proj)):
                for t in tqdm(range(num_classes)):
                    weq_t_proj[t] = sp_project(x_adv_t[t], weq_t[t], **def_params)
                for i in seq_images:
                    l = labels[i]
                    for t in range(num_classes):
                        x_adv_t[t, i] = x_adv_t_current[t, i] + delta * np.sign(weq_t_proj[t, i] - weq_t_proj[l, i])
                x_adv_t = np.clip(x_adv_t, x_tiled - eps, x_tiled + eps)
                x_adv_t = np.clip(x_adv_t, clip_min, clip_max)

        if progress is True:
            seq_targets_2 = tqdm(range(num_classes))

        for t in seq_targets_2:
            for i in seq_batches:
                batch = slice(i*batch_size, (i+1)*batch_size)
                logits_all[batch] = self.sess.run(logits, feed_dict={x_inp: x_adv_t[t, batch]})
            logits_t[:, t] = logits_all[:, t]
            for i in seq_images:
                l = labels[i]
                logits_t[i, t] -= logits_all[i, l]
                logits_t[i, l] = -np.inf

        targets = logits_t.argmax(axis=1)
        for i in seq_images_2:
            x_adv[i] = x_adv_t[targets[i], i]

        return x_adv

    def run_debug_single(self, x, num_classes, l, t, eps = 0.2,
                                                     delta = 0.01,
                                                     niter = 1000,
                                                     clip_min = 0.,
                                                     clip_max = 1.):
        """
        Debug mode - works with a single image and target label

        Inputs:
            x           - Image to be perturbed. Must be a Numpy array of shape 
                          [1, num_rows, num_cols, num_channels]
            num_classes - Number of classes
            l           - Source label. Must be in the range [0, num_classes-1]
            t           - Target label. Must be in the range [0, num_classes-1]           
            eps         - Overall L-inf budget for the attack.
            delta       - L-inf budget for each attack iteration.
            niter       - Number of attack iterations.
            clip_min    - Minimum input component value.
            clip_max    - Maximum input component value.

        Output:
            x_adv       - Perturbed image. Numpy array of same shape as x.
            logits_iter - Contains history of logit values after each iteration. 
                          Numpy array of size [niter, num_classes]
            e_iter      - Contains history of attacks after each iteration.
                          Numpy array of size [niter, 1, num_rows, num_cols, num_channels]
        """

        num_rows, num_cols, num_channels = x.shape[1:]

        # Create graph
        x_inp = tf.placeholder(tf.float32, shape=(1, num_rows, num_cols, num_channels))
        logits = self.model.get_logits(x_inp)
        grad_t, = tf.gradients(logits[:, t], x_inp)
        grad_l, = tf.gradients(logits[:, l], x_inp)

        x_adv = x.copy()
        logits_iter = np.zeros(niter, num_classes)
        e_iter = np.zeros(niter, 1, num_rows, num_cols, num_channels)

        for n in trange(niter):
            weq_t, weq_l, logits_iter[n] = self.sess.run([grad_t, weq_l, grad_l], feed_dict={x_inp: x_adv})
            x_adv += delta * np.sign(weq_t - weq_l)
            x_adv = np.clip(x_adv, x - eps, x + eps)
            x_adv = np.clip(x_adv, clip_min, clip_max)
            e_iter[n] = x_adv - x

        return x_adv, logits_iter, e_iter
        
    def run_binary(self, x, y, batch_size, eps = 0.2,
                                           delta = 0.01,
                                           niter = 1000,
                                           clip_min = 0.,
                                           clip_max = 1.,
                                           progress = True):
        """
        IterLocLinearAttack for binary (sigmoid) classification with fully connected layers

        Inputs:
            x           - Images to be perturbed. Must be a Numpy array of shape 
                          [num_images, dim]
            y           - True labels, binarized. Must be a Numpy array of shape
                          [num_images]
            eps         - Overall L-inf budget for the attack.
            delta       - L-inf budget for each attack iteration.
            niter       - Number of attack iterations.
            clip_min    - Minimum input component value.
            clip_max    - Maximum input component value.
            progress    - Whether to display progress bars

        Output:
            x_adv       - Perturbed image. Numpy array of same shape as x.
        """

        num_images, dim = x.shape

        # Create graph
        x_inp = tf.placeholder(tf.float32, shape=(None, dim))
        logits = self.model.get_logits(x_inp)
        grad, = tf.gradients(logits, x_inp)

        seq_batches = range(np.int(num_images/batch_size))
        seq_niter = range(niter)
        if progress is True:
            seq_niter = tqdm(seq_niter)

        weq = np.zeros([num_images, dim])

        sgn_y = 1.0 - 2.0*y
        x_adv = x.copy()
        for n in seq_niter:
            for i in seq_batches:
                batch = slice(i*batch_size, (i+1)*batch_size)
                weq[batch] = self.sess.run(grad, feed_dict={x_inp: x_adv[batch]})

            x_adv += delta * np.sign(weq) * sgn_y
            x_adv = np.clip(x_adv, x - eps, x + eps)
            x_adv = np.clip(x_adv, clip_min, clip_max)

        return x_adv

    def run_binary_proj(self, x, y, batch_size, niter_proj, 
                                                def_params,
                                                eps = 0.2,
                                                delta = 0.01,
                                                niter = 1000,
                                                clip_min = 0.,
                                                clip_max = 1.,
                                                progress = True):
        """
        IterLocLinearAttack for binary (sigmoid) classification with fully connected layers, 
        using iterated projections for gradient of front end

        Inputs:
            x           - Images to be perturbed. Must be a Numpy array of shape 
                          [num_images, dim]
            y           - True labels, binarized. Must be a Numpy array of shape
                          [num_images]
            niter_proj  - No of iterations for projection refinement
            def_params  - Defense parameters to pass to sp_project
            eps         - Overall L-inf budget for the attack.
            delta       - L-inf budget for each attack iteration.
            niter       - Number of attack iterations.
            clip_min    - Minimum input component value.
            clip_max    - Maximum input component value.
            progress    - Whether to display progress bars

        Output:
            x_adv       - Perturbed image. Numpy array of same shape as x.
        """

        num_images, dim = x.shape

        # Create graph
        x_inp = tf.placeholder(tf.float32, shape=(None, dim))
        logits = self.model.get_logits(x_inp)
        grad, = tf.gradients(logits, x_inp)

        seq_batches = range(np.int(num_images/batch_size))
        seq_niter = range(niter)
        if progress is True:
            seq_niter = tqdm(seq_niter)

        weq = np.zeros([num_images, dim])

        sgn_y = 1.0 - 2.0*y
        x_adv = x.copy()

        for i in seq_batches:
            batch = slice(i*batch_size, (i+1)*batch_size)
            weq[batch] = self.sess.run(grad, feed_dict={x_inp: x_adv[batch]})
        x_adv += delta * np.sign(weq) * sgn_y
        x_adv = np.clip(x_adv, x - eps, x + eps)
        x_adv = np.clip(x_adv, clip_min, clip_max)

        for n in seq_niter:
            x_adv_current = x_adv.copy()
            for i in seq_batches:
                batch = slice(i*batch_size, (i+1)*batch_size)
                weq[batch] = self.sess.run(grad, feed_dict={x_inp: x_adv[batch]})
            x_adv = x_adv_current + delta * np.sign(weq) * sgn_y
            x_adv = np.clip(x_adv, x - eps, x + eps)
            x_adv = np.clip(x_adv, clip_min, clip_max)

            for j in tqdm(range(niter_proj)):
                weq_proj = sp_project_binary(x_adv, weq, **def_params)
                x_adv = x_adv_current + delta * np.sign(weq) * sgn_y
                x_adv = np.clip(x_adv, x - eps, x + eps)
                x_adv = np.clip(x_adv, clip_min, clip_max)

        return x_adv


class FGSMAttack:
    """
    Fast Gradient Sign Method attack (Goodfellow et al, 2015)
    """
    def __init__(self, model, sess):
        """
        Inputs:
            model   - An instance of the cleverhans.model.Model class.
            sess    - The tf.Session to run graphs in.
        """
        
        if not isinstance(model, Model):
            raise TypeError('The model argument should be an instance of the cleverhans.model.Model class.')

        self.model = model
        self.sess = sess

    def run_proj(self, x, y, batch_size, niter_proj, 
                                         def_params,
                                         eps = 0.2,
                                         clip_min = 0.,
                                         clip_max = 1.,
                                         progress = True):
        """
        FGSM with iterated projections for gradient of front end

        Inputs:
            x           - Images to be perturbed. Must be a Numpy array of shape 
                          [num_images, num_rows, num_cols, num_channels]
            y           - True labels, one-hot encoded. Must be a Numpy array of shape
                          [num_images, num_classes]
            batch_size  - Number of attacks to run simultaneously.
            niter_proj  - No of iterations for projection refinement
            def_params  - Defense parameters to pass to sp_project
            eps         - Overall L-inf budget for the attack.
            clip_min    - Minimum input component value.
            clip_max    - Maximum input component value.
            progress    - Whether to display progress bars

        Output:
            x_pgd   - Perturbed images. Numpy array of same shape as x.

        """

        num_images, num_rows, num_cols, num_channels = x.shape
        num_classes = y.shape[1]

        if num_images % batch_size != 0:
            raise ValueError('Batch size must be a submultiple of data size')

        # Create graph
        x_inp = tf.placeholder(tf.float32, shape=(batch_size, num_rows, num_cols, num_channels))
        y_true = tf.placeholder(tf.float32, shape=(batch_size, num_classes))
        y_true = tf.stop_gradient(y_true)
        logits = self.model.get_logits(x_inp)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_true)
        gradient, = tf.gradients(loss, x_inp)

        x_fgsm = x.copy()

        seq_batches = range(np.int(num_images/batch_size))
        if progress is True:
            seq_batches = tqdm(seq_batches)

        for i in seq_batches:
            batch = slice(i*batch_size, (i+1)*batch_size)
            grad_batch = self.sess.run(gradient, feed_dict={x_inp: x_fgsm[batch], 
                                                                y_true: y[batch]})
            x_fgsm[batch] = x[batch] + eps * np.sign(grad_batch)
            x_fgsm[batch] = np.clip(x_fgsm[batch], clip_min, clip_max)
            for n in tqdm(range(niter_proj)):
                grad_batch_proj = sp_project(x_fgsm[batch], grad_batch, **def_params)
                x_fgsm[batch] = x[batch] + eps * np.sign(grad_batch_proj)
                x_fgsm[batch] = np.clip(x_fgsm[batch], clip_min, clip_max)

        return x_fgsm


class PGDAttack:
    """
    Projected Gradient Descent attack (Madry et al, 2018)
    """
    def __init__(self, model, sess):
        """
        Inputs:
            model   - An instance of the cleverhans.model.Model class.
            sess    - The tf.Session to run graphs in.
        """
        
        if not isinstance(model, Model):
            raise TypeError('The model argument should be an instance of the cleverhans.model.Model class.')

        self.model = model
        self.sess = sess

    def run_multiple(self, x, y, batch_size, 
                                 num_runs = 100,
                                 eps = 0.2,
                                 delta = 0.05,
                                 niter = 100,
                                 clip_min = 0.,
                                 clip_max = 1.,
                                 rand_init = True,
                                 rand_seed = None,
                                 progress = True,
                                 verbose = True):
        """
        Run multiple runs of PGD with different random initializations, and report the 
        accuracy over the most successful run(s) for *each* image.

        For a faster attack, set batch_size > num_images. Note that batch_size needs to 
        be a submultiple of num_runs*num_images. 

        Inputs:
            x           - Images to be perturbed. Must be a Numpy array of shape 
                          [num_images, num_rows, num_cols, num_channels].
            y           - True labels, one-hot encoded. Must be a Numpy array of shape
                          [num_images, num_classes].
            batch_size  - Number of attacks to run simultaneously. This must be a 
                          submultiple of num_images*num_runs.
            num_runs    - Number of runs with different random initializations.
            eps         - Overall L-inf budget for the attack.
            delta       - L-inf budget for each attack iteration.
            niter       - Number of attack iterations.
            clip_min    - Minimum input component value.
            clip_max    - Maximum input component value.
            rand_init   - Whether to use random initialization.
            rand_seed   - Seed for random number generator.
            progress    - Whether to display progress bars

        Output:
            x_pgd       - Perturbed images. Numpy array of shape
                          [num_runs, num_images, num_rows, num_cols, num_channels]

        """

        num_images, num_rows, num_cols, num_channels = x.shape
        num_classes = y.shape[1]

        if (num_images*num_runs) % batch_size != 0:
            raise ValueError('Batch size must be a submultiple of num_images*num_runs ')

        if rand_init is False:
            raise ValueError('Cannot run multiple runs without random initialization')

        # Create graph
        x_inp = tf.placeholder(tf.float32, shape=(None, num_rows, num_cols, num_channels))
        y_true = tf.placeholder(tf.float32, shape=(None, num_classes))
        y_true = tf.stop_gradient(y_true)
        logits = self.model.get_logits(x_inp)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_true)
        gradient, = tf.gradients(loss, x_inp)

        x_tiled = np.tile(x, [num_runs, 1, 1, 1])
        y_tiled = np.tile(y, [num_runs, 1])

        rng = RandomState(rand_seed)
        x_pgd = x_tiled + rng.uniform(low=-eps, high=eps, size=x_tiled.shape)
        x_pgd = np.clip(x_pgd, clip_min, clip_max)

        seq_batches = range(np.int(num_images*num_runs/batch_size))
        seq_niter = range(niter)

        if progress is True:
            seq_batches = tqdm(seq_batches)
        for i in seq_batches:
            batch = slice(i*batch_size, (i+1)*batch_size)
            if progress is True:
                seq_niter = tqdm(range(niter))
            for n in seq_niter:
                grad_batch = self.sess.run(gradient, feed_dict={x_inp: x_pgd[batch], 
                                                                y_true: y_tiled[batch]})
                x_pgd[batch] += delta * np.sign(grad_batch)
                x_pgd[batch] = np.clip(x_pgd[batch], x_tiled[batch] - eps, x_tiled[batch] + eps)
                x_pgd[batch] = np.clip(x_pgd[batch], clip_min, clip_max)

        if verbose is True:
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_true, 1))
            seq_batches = range(np.int(num_images*num_runs/batch_size))
            preds = np.zeros([num_runs*num_images])
            for i in seq_batches:
                batch = slice(i*batch_size, (i+1)*batch_size)
                preds[batch] = self.sess.run(correct_prediction, 
                                             feed_dict={x_inp: x_pgd[batch], 
                                                        y_true: y_tiled[batch]})
            preds = preds.reshape([num_runs, num_images])
            acc = 100*preds.min(axis=0).mean()
            print('PGD accuracy: {:.2f}%\n'.format(acc))

        x_pgd = x_pgd.reshape([num_runs, num_images, num_rows, num_cols, num_channels])

        return x_pgd

    def run_multiple_proj(self, x, y, batch_size, niter_proj, def_params,
                                            num_runs = 100,
                                            eps = 0.2,
                                            delta = 0.05,
                                            niter = 100,
                                            clip_min = 0.,
                                            clip_max = 1.,
                                            rand_init = True,
                                            rand_seed = None,
                                            progress = True,
                                            verbose = True):
        """
        PGD with multiple random restarts, using iterated projections for gradient of front end

        Inputs:
            x           - Images to be perturbed. Must be a Numpy array of shape 
                          [num_images, num_rows, num_cols, num_channels]
            y           - True labels, one-hot encoded. Must be a Numpy array of shape
                          [num_images, num_classes]
            batch_size  - Number of attacks to run simultaneously.
            niter_proj  - No of iterations for projection refinement
            def_params  - Defense parameters to pass to sp_project
            num_runs    - Number of runs with different random initializations
            eps         - Overall L-inf budget for the attack.
            delta       - L-inf budget for each attack iteration.
            niter       - Number of attack iterations.
            clip_min    - Minimum input component value.
            clip_max    - Maximum input component value.
            rand_init   - Whether to use random initialization.
            rand_seed   - Seed for random number generator.
            progress    - Whether to display progress bars

        Output:
            x_pgd       - Perturbed images. Numpy array of shape
                          [num_images, num_rows, num_cols, num_channels]

        """

        num_images, num_rows, num_cols, num_channels = x.shape
        num_classes = y.shape[1]

        if (num_images*num_runs) % batch_size != 0:
            raise ValueError('Batch size must be a submultiple of num_images*num_runs ')

        if rand_init is False:
            raise ValueError('Cannot run multiple runs without random initialization')

        # Create graph
        x_inp = tf.placeholder(tf.float32, shape=(None, num_rows, num_cols, num_channels))
        y_true = tf.placeholder(tf.float32, shape=(None, num_classes))
        y_true = tf.stop_gradient(y_true)
        logits = self.model.get_logits(x_inp)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_true)
        gradient, = tf.gradients(loss, x_inp)

        x_tiled = np.tile(x, [num_runs, 1, 1, 1])
        y_tiled = np.tile(y, [num_runs, 1])

        rng = RandomState(rand_seed)
        x_pgd = x_tiled + rng.uniform(low=-eps, high=eps, size=x_tiled.shape)
        x_pgd = np.clip(x_pgd, clip_min, clip_max)

        # seq_batches = range(np.int(num_images/batch_size))
        seq_batches = range(np.int(num_images*num_runs/batch_size))
        seq_niter = range(niter)

        x_pgd_current = x_pgd.copy()
        if progress is True:
            seq_batches = tqdm(seq_batches)
        for i in seq_batches:
            batch = slice(i*batch_size, (i+1)*batch_size)
            if progress is True:
                seq_niter = tqdm(range(niter))
            for n in seq_niter:
                x_pgd_current[batch] = x_pgd[batch].copy()
                grad_batch = self.sess.run(gradient, feed_dict={x_inp: x_pgd[batch], 
                                                                y_true: y_tiled[batch]})
                x_pgd[batch] += delta * np.sign(grad_batch)
                x_pgd[batch] = np.clip(x_pgd[batch], x_tiled[batch] - eps, x_tiled[batch] + eps)
                x_pgd[batch] = np.clip(x_pgd[batch], clip_min, clip_max)
                for j in tqdm(range(niter_proj)):
                    grad_batch_proj = sp_project(x_pgd[batch], grad_batch, **def_params)
                    x_pgd[batch] = x_pgd_current[batch] + delta * np.sign(grad_batch_proj)
                    x_pgd[batch] = np.clip(x_pgd[batch], x_tiled[batch] - eps, x_tiled[batch] + eps)
                    x_pgd[batch] = np.clip(x_pgd[batch], clip_min, clip_max)

        if verbose is True:
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_true, 1))
            seq_batches = range(np.int(num_images*num_runs/batch_size))
            preds = np.zeros([num_runs*num_images])
            for i in seq_batches:
                batch = slice(i*batch_size, (i+1)*batch_size)
                preds[batch] = self.sess.run(correct_prediction, 
                                             feed_dict={x_inp: x_pgd[batch], 
                                                        y_true: y_tiled[batch]})
            preds = preds.reshape([num_runs, num_images])
            acc = 100*preds.min(axis=0).mean()
            print('PGD accuracy: {:.2f}%\n'.format(acc))

        x_pgd = x_pgd.reshape([num_runs, num_images, num_rows, num_cols, num_channels])

        return x_pgd

    def run_debug(self, x, y, batch_size, eps = 8.,
                                    delta = 1.,
                                    niter = 100,
                                    clip_min = 0.,
                                    clip_max = 255.,
                                    rand_init = False,
                                    rand_seed = None,
                                    progress = True):
        """
        Inputs:
            x           - Images to be perturbed. Must be a Numpy array of shape 
                          [num_images, num_rows, num_cols, num_channels]
            y           - True labels, one-hot encoded. Must be a Numpy array of shape
                          [num_images, num_classes]
            batch_size  - Number of attacks to run simultaneously.
            eps         - Overall L-inf budget for the attack.
            delta       - L-inf budget for each attack iteration.
            niter       - Number of attack iterations.
            clip_min    - Minimum input component value.
            clip_max    - Maximum input component value.
            rand_init   - Whether to use random initialization.
            rand_seed   - Seed for random number generator.
            progress    - Whether to display progress bars

        Output:
            x_pgd   - Perturbed images. Numpy array of same shape as x.

        """

        num_images, num_rows, num_cols, num_channels = x.shape
        num_classes = y.shape[1]

        if num_images % batch_size != 0:
            raise ValueError('Batch size must be a submultiple of data size')

        # Create graph
        x_inp = tf.placeholder(tf.float32, shape=(batch_size, num_rows, num_cols, num_channels))
        y_true = tf.placeholder(tf.float32, shape=(batch_size, num_classes))
        y_true = tf.stop_gradient(y_true)
        logits = self.model.get_logits(x_inp)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_true)
        gradient, = tf.gradients(loss, x_inp)

        if rand_init is True:
            rng = RandomState(rand_seed)
            x_pgd = x + rng.uniform(low=-eps, high=eps, size=x.shape)
            x_pgd = np.clip(x_pgd, clip_min, clip_max)
        else:
            x_pgd = x.copy()

        x_pgd_iter = np.zeros([niter, num_images, num_rows, num_cols, num_channels])

        seq_batches = range(np.int(num_images/batch_size))
        seq_niter = range(niter)
        if progress is True:
            seq_batches = tqdm(seq_batches)

        for i in seq_batches:
            batch = slice(i*batch_size, (i+1)*batch_size)
            if progress is True:
                seq_niter = tqdm(range(niter))
            for n in seq_niter:
                grad_batch = self.sess.run(gradient, feed_dict={x_inp: x_pgd[batch], 
                                                                y_true: y[batch]})
                x_pgd[batch] += delta * np.sign(grad_batch)
                x_pgd[batch] = np.clip(x_pgd[batch], x[batch] - eps, x[batch] + eps)
                x_pgd[batch] = np.clip(x_pgd[batch], clip_min, clip_max)
                x_pgd_iter[n, batch] = x_pgd[batch]

        return x_pgd, x_pgd_iter