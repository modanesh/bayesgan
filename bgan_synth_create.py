#!/usr/bin/env python

import os
import sys

import tensorflow as tf
import numpy as np

from bgan_util import SynthDataset, SynthDataset_generated, FigPrinter
from collections import OrderedDict, defaultdict

from bgan_util import AttributeDict

from dcgan_ops import *

from sklearn import mixture


def get_session():
    global _SESSION
    if tf.get_default_session() is None:
        _SESSION = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
    else:
        _SESSION = tf.get_default_session()

    return _SESSION

class BGAN_create(object):

    def __init__(self, x_dim, z_dim, dataset_size, batch_size=64, prior_std=1.0, num_classes=1, alpha=0.01,
                 optimizer='adam', ml=False):

        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.optimizer = optimizer.lower()

        # Bayes
        self.prior_std = prior_std
        self.alpha = alpha
   
        self.weight_dims = OrderedDict([("g_h0_lin_W", (self.z_dim, 1000)),
                                        ("g_h0_lin_b", (1000,)),
                                        ("g_lin_W", (1000, self.x_dim[0])),
                                        ("g_lin_b", (self.x_dim[0],))])
        
        self.K = num_classes # 1 means unsupervised, label == 0 always reserved for fake

        self.build_bgan_graph()

    def build_bgan_graph(self):
    
        self.inputs = tf.placeholder(tf.float32,
                                     [self.batch_size] + self.x_dim, name='real_images')
        
        self.labeled_inputs = tf.placeholder(tf.float32,
                                             [self.batch_size] + self.x_dim, name='real_images_w_labels')
        
        self.labels = tf.placeholder(tf.float32,
                                     [self.batch_size, self.K+1], name='real_targets')


        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        #self.z_sum = histogram_summary("z", self.z) TODO looks cool

        ### Generator
        self.gen_param_list = []
        with tf.variable_scope("generator") as scope:
            gen_params = AttributeDict()
            for name, shape in self.weight_dims.items():
                gen_params[name] = tf.get_variable("%s" % (name),
                                                   shape, initializer=tf.random_normal_initializer(stddev=0.02))
            self.gen_param_list.append(gen_params)

  

        self.generation = {}
        for gen_params in self.gen_param_list:
            self.generation["g_prior"]=self.gen_prior(gen_params)
            self.generation["generators"]=self.generator(self.z, gen_params)
            self.generation["gen_samplers"]=self.sampler(self.z, gen_params)

            
    def generator(self, z, gen_params):
        with tf.variable_scope("generator") as scope:
            h0 = lrelu(linear(z, 1000, 'g_h0_lin',
                              matrix=gen_params.g_h0_lin_W, bias=gen_params.g_h0_lin_b))
            self.x_ = linear(h0, self.x_dim[0], 'g_lin',
                             matrix=gen_params.g_lin_W, bias=gen_params.g_lin_b)
            return self.x_

    def sampler(self, z, gen_params):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
            return self.generator(z, gen_params)

    def gen_prior(self, gen_params):
        with tf.variable_scope("generator") as scope:
            prior_loss = 0.0
            for var in gen_params.values():
                print(var)
                nn = tf.divide(var, self.prior_std)
                prior_loss += tf.reduce_mean(tf.multiply(nn, nn))
                
        prior_loss /= self.dataset_size

        return prior_loss

    

def bgan_synth(x_dim, z_dim, batch_size=64, numz=5, num_iter=1000, rpath="synth_results",
               base_learning_rate=1e-2, lr_decay=3., save_weights=False, num_classes=1, num_iter_test=0):

    bgan = BGAN_create([x_dim], z_dim,
                 num_iter,
                batch_size=batch_size,
                prior_std=10.0, alpha=1e-3, 
                num_classes=num_classes,
                )
    
    
    print ("Starting session")
    session = get_session()
    
    tf.global_variables_initializer().run()

    print ("Starting training loop")
    mean = np.random.uniform(-1, 1, size=z_dim)
    cov = np.random.uniform(-1, 1, size=(z_dim,z_dim))
    cov = np.dot(cov,cov.transpose())
    
    count = 0
    for train_iter in range(int(num_iter/batch_size) + 1):
        if count + batch_size < num_iter:
            #sample_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))
            sample_z = np.random.multivariate_normal(mean, cov, size=batch_size)
            #sample_z = np.random.normal(0, 1.0, size=(batch_size, z_dim))
        else:
            #sample_z = np.random.uniform(-1, 1, size=(num_iter-count, z_dim))
            sample_z = np.random.multivariate_normal(mean, cov, size=num_iter-count)
            #sample_z = np.random.normal(0, 1.0, size=(num_iter-count, z_dim))
            
        n, _ = sample_z.shape

        count += n
        sampled_data = session.run(bgan.generation["gen_samplers"], feed_dict={bgan.z: sample_z})
        if train_iter == 0:
            fake_data_x = sampled_data
            fake_data_z = sample_z
        else:
            fake_data_x = tf.concat([fake_data_x, sampled_data], 0)
            fake_data_z = tf.concat([fake_data_z, sample_z], 0)

        print("Count {}".format(count))

    count = 0
    for train_iter in range(int(num_iter_test/batch_size) + 1):
        if count + batch_size < num_iter:
            #sample_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))
            sample_z = np.random.multivariate_normal(mean, cov, size=batch_size)
            #sample_z = np.random.normal(0, 1.0, size=(batch_size, z_dim))
        else:
            #sample_z = np.random.uniform(-1, 1, size=(num_iter-count, z_dim))
            sample_z = np.random.multivariate_normal(mean, cov, size=num_iter-count)
            #sample_z = np.random.normal(0, 1.0, size=(num_iter-count, z_dim))
            
        n, _ = sample_z.shape

        count += n
        sampled_data = session.run(bgan.generation["gen_samplers"], feed_dict={bgan.z: sample_z})
        if train_iter == 0:
            fake_data_x_test = sampled_data
            fake_data_z_test = sample_z
        else:
            fake_data_x_test = tf.concat([fake_data_x, sampled_data], 0)
            fake_data_z_test = tf.concat([fake_data_z, sample_z], 0)

        print("Count Test {}".format(count))    
        
        
    if save_weights:
        var_dict = {}
        for var in tf.trainable_variables():
            var_dict[var.name] = session.run(var.name)

        np.savez_compressed(os.path.join(rpath,
                                         "weights_%i.npz" % train_iter),
                            **var_dict)

    return {"dataset": fake_data_x.eval(),
            "dataset_test": fake_data_x_test.eval(),
            "noise_z" : fake_data_z.eval(),
            "z_dim": z_dim,
            "x_dim" : x_dim,
            "num_iter": num_iter,
            "num_iter_test": num_iter_test}


if __name__ == "__main__":

    import argparse
    import time

    parser = argparse.ArgumentParser(description='Script to run Bayesian GAN synthetic experiments')

    parser.add_argument('--x_dim',
                        type=int,
                        default=500,
                        help='dim of x for synthetic data')
    parser.add_argument('--z_dim',
                        type=int,
                        default=10,
                        help='dim of z for generator')
    parser.add_argument('--num_samples_gen',
                        type=int,
                        default=10000,
                        help='no of samples to generate for the dataset')
    parser.add_argument('--num_samples_gen_test',
                        type=int,
                        default=5000,
                        help='no of samples to generate for the test dataset')
    parser.add_argument('--out_dir',
                        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "Result/"),
                        help='path of where to store results')
    parser.add_argument('--random_seed',
                        type=int,
                        default=2222,
                        help='set seed for repeatability')
    parser.add_argument('--num_classes',
                        type=int,
                        default=1,
                        help='Number of classes')
    args = parser.parse_args()

    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        tf.set_random_seed(args.random_seed)

    if not os.path.exists(args.out_dir):
        print ("Creating %s" % args.out_dir)
        os.makedirs(args.out_dir)

    results_path = os.path.join(args.out_dir, "experiment_%i_gen" % (int(time.time())))
    os.makedirs(results_path)
    import pprint
    with open(os.path.join(results_path, "args.txt"), "w") as hf:
        hf.write("Experiment settings:\n")
        hf.write("%s\n" % (pprint.pformat(args.__dict__)))
    
    # Training set
    results = bgan_synth(args.x_dim, args.z_dim, num_iter=args.num_samples_gen, num_classes=args.num_classes,
                         rpath=results_path, save_weights=True, num_iter_test=args.num_samples_gen_test)

    np.savez(os.path.join(results_path, "run_regular_bayes.npz"),
             **results)

    




