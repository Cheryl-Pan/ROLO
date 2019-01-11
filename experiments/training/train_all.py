# Copyright (c) <2016> <GUANGHAN NING>. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



'''
Script File: ROLO_step6_train_20_exp1.py

Description:

	ROLO is short for Recurrent YOLO, aimed at simultaneous object detection and tracking
	Paper: http://arxiv.org/abs/1607.05781
	Author: Guanghan Ning
	Webpage: http://guanghan.info/
'''

# !/usr/bin/env python

# Imports
import os,sys


abspath = os.path.abspath("..")  # ~/ROLO/experiments
rootpath = os.path.split(abspath)[0]  # ~/ROLO
sys.path.append(rootpath)
from utils import ROLO_utils as utils

import tensorflow as tf
# from tensorflow.models.rnn import rnn, rnn_cell
import cv2

import numpy as np
import os
import os.path
import time
import random

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"


class ROLO_TF:
    disp_console = False  # False
    restore_weights = False  # True

    # YOLO parameters
    fromfile = None
    tofile_img = 'test/output.jpg'
    tofile_txt = 'test/output.txt'
    imshow = True
    filewrite_img = False
    filewrite_txt = False
    yolo_weights_file = 'weights/YOLO_small.ckpt'
    alpha = 0.1
    threshold = 0.2
    iou_threshold = 0.5
    num_class = 20
    num_box = 2
    grid_size = 7
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    w_img, h_img = [352, 240]

    # ROLO Network Parameters
    rolo_model_file = 'panchen/output/ROLO_model_2'
    rolo_weights_file = os.path.join(rolo_model_file, 'model_step6_exp1.ckpt')
    lstm_depth = 3
    num_steps = 6  # number of frames as an input sequence
    num_feat = 4096
    num_predict = 6  # final output of LSTM 6 loc parameters
    num_unit = 4096+6
    num_gt = 4
    num_input = num_feat + num_predict  # data input: 4096+6= 4012

    # ROLO Training Parameters
    # learning_rate = 0.00001 #training
    learning_rate = 0.0001  # testing

    training_iters = 210  # 100000
    batch_size = 1  # a kind of piceture only have one
    display_step = 1

    # tf Graph input
    x = tf.placeholder(tf.float32, shape=[None, num_steps, num_input])
    print (x.shape)

    y = tf.placeholder(tf.float32, [None, num_gt])

    # Define weights
    # with tf.variable_scope("weight", reuse=True):
    #     weights = {
    #         'out': tf.Variable(tf.random_normal([num_input, num_gt]))
    #     }
    # with tf.variable_scope("bias", reuse=True):
    #     biases = {
    #         'out': tf.Variable(tf.random_normal([num_gt]))
    #     }

    def __init__(self, argvs=None):
        if argvs is None:
            argvs = []
        print("ROLO init")
        self.ROLO(argvs)
        # self.LSTM_single("lstm",self.x,self.istate,self.weights,self.biases)
        # self.lstm_single_2(self.x)

    def LSTM_single(self, name, _X, _istate, _weights, _biases):
        print (_X.shape)
        # input shape: (batch_size, n_steps, n_input) (?,6,4102)
        _X = tf.transpose(_X, [1, 0, 2])  # permute num_steps and batch_size _X.shape:(n_steps,batch_size,n_input)
        print (_X.shape)
        # Reshape to prepare input to hidden activation
        _X = tf.reshape(_X, [self.num_steps * self.batch_size, self.num_input])  # (num_steps*batch_size, num_input)
        print (_X.shape)
        # Split data because rnn cell needs a list of inputs for the RNN inner loop
        _X = tf.split(_X, self.num_steps, 0)  # n_steps * (batch_size, num_input)
        # print("_X: ", _X)
        print _X

        # cell = tf.nn.rnn_cell.LSTMCell(self.num_input) #BasicLSTMCell 4102
        cell = tf.nn.rnn_cell.LSTMCell(self.num_input, self.num_input)
        state = _istate
        # outputs, states  = tf.nn.dynamic_rnn(cell, _X, initial_state=state, time_major = True)
        # tf.get_variable_scope().reuse_variables()
        with tf.variable_scope('lstm_rnn'):
            for step in range(self.num_steps):
                outputs, state = tf.nn.static_rnn(cell, [_X[step]], dtype=tf.float32)  # _X[step]:(batch_size,num_input)
                tf.get_variable_scope().reuse_variables()

        # print("output: ", outputs)
        # print("state: ", state)
        return outputs  # outputis a list,include one tensor, TensorShape:[batch_size, num_input]


    def lstm_single_2(self,name, x_input):
        x_in = tf.transpose(x_input, [1, 0, 2])  # [n_step, batch_size, num_input]
        print x_in.shape
        print '\n'
        x_in = tf.reshape(x_in,[self.num_steps * self.batch_size, self.num_input])
        print x_in.shape
        print '\n'
        x_reverse = tf.reverse(x_in,axis=[0])
        x_in_fw = tf.split(x_in, self.num_steps, 0)
        x_in_bw = tf.split(x_reverse,self.num_steps,0)

        lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=self.num_unit, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=self.num_unit, forget_bias=1.0, state_is_tuple=True)

        out_list_fw = []
        out_list_bw = []
        with tf.variable_scope('bidirectional_lstm',reuse=tf.AUTO_REUSE)as scope:
            # forward direction
            for step in range(self.num_steps):
                if step > 0:
                    tf.get_variable_scope().reuse_variables()
                outputs_fw, _ = tf.nn.static_rnn(lstm_cell_fw,[x_in_fw[step]],dtype=tf.float32)
                outputs_bw, _ = tf.nn.static_rnn(lstm_cell_bw,[x_in_bw[step]],dtype=tf.float32)
                out_list_fw.append(outputs_fw[0])
                out_list_bw.append(outputs_bw[0])
            out_list_bw = list(reversed(out_list_bw))
            print 'len of out list fw\n',len(out_list_fw)
            print 'len of out list bw\n',len(out_list_bw)
            fw_pred = out_list_fw[-1][:,4097:4101]
            bw_pred = out_list_bw[-1][:,4097:4101]
            final_out = tf.add(fw_pred,bw_pred)/2
        return fw_pred,bw_pred,final_out

    def bi_lstm_2(self, name, X):
        x_in = tf.transpose(X, [1, 0, 2])  # [n_step,batch_size,num_input]
        lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(self.num_input)
        lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(self.num_input)

        output_bi_lstm, states = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, x_in, dtype=tf.float32,
                                                                 time_major=True)
        # output_bi_lstm is a tuple,(output_fw,output_bw)
        output_fw_pred = output_bi_lstm[0][-1][:, 4097:4101]
        output_bw_pred = output_bi_lstm[1][-1][:, 4097:4101]

        final_output = tf.add(output_fw_pred, output_bw_pred) / 2  # shape:(batch_size,4)
        # outputs = tf.concat(output_bi_lstm,2) #(n_step,batchsize,num_input*2)
        # outputs[-1][:,4097:4101]
        return final_output  # last time_step output,(1,tensor([batch_size,num_input*2]))

    # Experiment with dropout
    def dropout_features(self, feature, prob):
        num_drop = int(prob * 4096)
        drop_index = random.sample(xrange(4096), num_drop)
        for i in range(len(drop_index)):
            index = drop_index[i]
            feature[index] = 0
        return feature

    '''---------------------------------------------------------------------------------------'''

    def build_networks(self):
        if self.disp_console: print "Building ROLO graph..."

        # Build rolo layers
        # self.lstm_module = self.LSTM_single('lstm_test', self.x, self.istate, self.weights, self.biases)
        self.lstm_module = self.bi_lstm("bi_lstm", self.x)
        # self.lstm_module = self.bi_lstm_2('bi_lstm_2',self.x)
        # self.ious = tf.Variable(tf.zeros([self.batch_size]), name="ious")
        # self.sess = tf.Session()
        # self.sess.run(tf.global_variables_initializer())
        # self.saver = tf.train.Saver()
        # self.saver.restore(self.sess, self.rolo_weights_file)
        if self.disp_console: print "Loading complete!" + '\n'


    def train_20(self):
        print("TRAINING ROLO...")
        # print ("build_network...")
        # self.build_networks()

        ''' TUNE THIS'''
        num_videos = 22
        epoches = 22 * 100   # 20 * 100

        # Use rolo_input for LSTM training
        fw_pred, bw_pred, pred = self.lstm_single_2("bi_lstm",self.x)
        with tf.name_scope('loss'):
            correct_prediction = tf.square(pred - self.y)
            lstm_pred = tf.square(fw_pred-bw_pred)
            self.accuracy = (tf.reduce_mean(correct_prediction)+tf.reduce_mean(lstm_pred)) * 100
            tf.summary.histogram('loss', self.accuracy)
        self.learning_rate = 0.00001 #0.00001
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.accuracy)  # Adam Optimizer

        merged_summary = tf.summary.merge_all()
        # Initializing the variables
        init = tf.global_variables_initializer()

        self.saver = tf.train.Saver(max_to_keep=1)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # Launch the graph
        with tf.Session(config=config) as sess:
            writer = tf.summary.FileWriter('log_train_all')
            writer.add_graph(sess.graph)
            sess.run(init)
            # ckpt = tf.train.get_checkpoint_state(self.rolo_model_file)
            # if ckpt and ckpt.model_checkpoint_path:
            #     print ckpt.model_checkpoint_path
            #     self.saver.restore(sess, ckpt.model_checkpoint_path)

            total_time = 0
            for epoch in range(epoches):  # 22*50
                log_file = open('panchen/output/training-20-log.txt', 'a')
                i = epoch % num_videos  # 22
                [self.w_img, self.h_img, sequence_name, dummy, self.training_iters] = utils.choose_video_sequence(i)

                x_path = os.path.join('../../benchmark/DATA', sequence_name, 'yolo_out/')
                y_path = os.path.join('../../benchmark/DATA', sequence_name, 'groundtruth_rect.txt')
                self.output_path = os.path.join('../../benchmark/DATA', sequence_name, 'rolo_out_train/')
                utils.createFolder(self.output_path)
                total_loss = 0
                id = 0

                start_time = time.time()

                # Keep training until reach max iterations
                while id < self.training_iters - self.num_steps + 1:
                    # print('id is %d:' % (id))
                    # Load training data & ground truth
                    batch_xs = self.rolo_utils.load_yolo_output_test(x_path, self.batch_size, self.num_steps, id)  # [num_of_examples, num_input] (depth == 1)


                    # Apply dropout to batch_xs
                    # for item in range(len(batch_xs)):
                    #    batch_xs[item] = self.dropout_features(batch_xs[item], 0.4)

                    # print(id)
                    batch_ys = self.rolo_utils.load_rolo_gt_test(y_path, self.batch_size, self.num_steps, id)
                    batch_ys = utils.locations_from_0_to_1(self.w_img, self.h_img, batch_ys)

                    # Reshape data to get 3 seq of 4102 elements
                    batch_xs = np.reshape(batch_xs, [self.batch_size, self.num_steps, self.num_input])
                    batch_ys = np.reshape(batch_ys, [self.batch_size, 4])
                    if self.disp_console: print("Batch_ys: ", batch_ys)

                    pred_location,loss, _ = sess.run([pred,self.accuracy,self.optimizer], feed_dict={self.x: batch_xs, self.y: batch_ys})
                    if self.disp_console: print("ROLO Pred: ", pred_location)
                    # print("len(pred) = ", len(pred_location))
                    utils.save_rolo_output(self.output_path, pred_location, id, self.num_steps, self.batch_size)

                    # print("correct_prediction int: ", (pred_location + 0.1).astype(int))

                    # Save pred_location to file
                    # sess.run(self.optimizer, feed_dict={self.x: batch_xs, self.y: batch_ys,self.istate: np.zeros((self.batch_size, 2 * self.num_input))})

                    # if id % self.display_step == 0:
                    #     # Calculate batch loss
                    #     loss = sess.run(self.accuracy, feed_dict={self.x: batch_xs, self.y: batch_ys,
                    #                                               self.istate: np.zeros(
                    #                                                   (self.batch_size, 2 * self.num_input))})
                    if self.disp_console:print ("Iter " + str(
                                id * self.batch_size) + ", Minibatch Loss= " + "{:.6f}".format(
                                loss))  # + "{:.5f}".format(self.accuracy)
                    total_loss += loss
                    id += 1
                    if self.disp_console: print(id)
                    if id % 1000 == 0:
                        summary = sess.run(merged_summary,feed_dict={self.x:batch_xs,self.y:batch_ys})
                        writer.add_summary(summary, id)

                cycle_time = time.time() - start_time
                print('video iteration is %d, video: %d time is %.2f ' % (epoch/22+1, epoch%22+1, cycle_time))
                log_file.write(str('video iteration is %d, video: %d \n' % (epoch/22+1, epoch%22+1)))
                total_time += cycle_time

                avg_loss = total_loss / id
                print "Avg loss: " + sequence_name + ": " + str(avg_loss)
                log_file.write(sequence_name+ "  avg loss is "+str(avg_loss)+'\n')

                if i + 1 == num_videos:
                    log_file.write('\n' + 'epoch is ' + str(epoch) + '\n')
                    log_file.write('total time: ' + str(total_time) + '\n')
                    print 'total_time is %.2f' % total_time
                    save_path = self.saver.save(sess, self.rolo_weights_file, global_step=epoch+1)
                    print ("Model saved in file: %s" % save_path)
                log_file.close()
        return

    def ROLO(self, argvs):

        self.rolo_utils = utils.ROLO_utils()
        self.rolo_utils.loadCfg()
        self.params = self.rolo_utils.params

        arguments = self.rolo_utils.argv_parser(argvs)

        if self.rolo_utils.flag_train is True:
            self.training(self.rolo_utils.x_path, self.rolo_utils.y_path)
        elif self.rolo_utils.flag_track is True:
            self.build_networks()
            self.track_from_file(self.rolo_utils.file_in_path)
        elif self.rolo_utils.flag_detect is True:
            self.build_networks()
            self.detect_from_file(self.rolo_utils.file_in_path)
        else:
            self.train_20()

    '''----------------------------------------main-----------------------------------------------------'''


def main(argvs):
    ROLO_TF(argvs)


if __name__ == '__main__':
    main(' ')
