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
import utils.ROLO_utils as utils

import tensorflow as tf
# from tensorflow.models.rnn import rnn, rnn_cell
import cv2

import numpy as np
import os.path
import time
import random
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


class ROLO_TF:
    disp_console = True  # False
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
    # rolo_weights_file = '/u03/Guanghan/dev/ROLO-dev/output/ROLO_model/model_step6_exp1.ckpt'
    rolo_weights_file = 'panchen/output/ROLO_model/model_step6_exp1.ckpt'
    lstm_depth = 3
    num_steps = 6  # number of frames as an input sequence
    num_feat = 4096
    num_predict = 6  # final output of LSTM 6 loc parameters
    num_unit = 3000
    num_gt = 4
    num_input = num_feat + num_predict  # data input: 4096+6= 4012

    # ROLO Training Parameters
    # learning_rate = 0.00001 #training
    learning_rate = 0.00001  # testing

    training_iters = 210  # 100000
    batch_size = 1  # a kind of piceture only have one
    display_step = 1

    # tf Graph input
    x = tf.placeholder(tf.float32, shape=[None, num_steps, num_input])
    print (x.shape)
    istate = tf.placeholder(tf.float32, shape=[None, 2 * num_input])  # state & cell => 2x num_input
    y = tf.placeholder(tf.float32, [None, num_gt])

    # Define weights
    with tf.variable_scope("weight", reuse=True):
        weights = {
            'in': tf.Variable(tf.random_normal([num_input, num_input]))
        }
    with tf.variable_scope("bias", reuse=True):
        biases = {
            'in': tf.Variable(tf.random_normal([num_input]))
        }

    def __init__(self, argvs=None):
        if argvs is None:
            argvs = []
        print("ROLO init")
        self.ROLO(argvs)

    def LSTM_single(self, name, _X, _istate, _weights, _biases):
        print (_X.shape)
        # input shape: (batch_size, n_steps, n_input) (?,6,4102)
        _X = tf.transpose(_X, [1, 0, 2])  # permute num_steps and batch_size _X.shape:(n_steps,batch_size,n_input)
        print (_X.shape)
        # Reshape to prepare input to hidden activation
        _X = tf.reshape(_X, [self.num_steps * self.batch_size, self.num_input])  # (num_steps*batch_size, num_input)
        # print (_X.shape)
        # Split data because rnn cell needs a list of inputs for the RNN inner loop
        _X = tf.split(_X, self.num_steps, 0)  # n_steps * (batch_size, num_input)
        # print("_X: ", _X)

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

    # bi-LSTM
    # def bi_lstm(self, name, X):
    #     _X = tf.reshape(X, [-1, self.num_input])
    #     w_in = self.weights['in']
    #     b_in = self.biases['in']
    #     _X = tf.matmul(_X, w_in) + b_in
    #     x_in = tf.reshape(_X, [-1, self.num_steps, self.num_input])
    #     x_in = tf.transpose(x_in, [1, 0, 2])  # [n_step,batch_size,num_input]
    #     with tf.variable_scope('forward'):
    #         lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(self.num_input)
    #     with tf.variable_scope('backward'):
    #         lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(self.num_input)
    #
    #     output_bi_lstm, states = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, x_in, dtype=tf.float32,
    #                                                              time_major=True)
    #     # output_bi_lstm is a tuple,(output_fw,output_bw)
    #     output_fw_pred = output_bi_lstm[0][-1][:, 4097:4101]
    #     output_bw_pred = output_bi_lstm[1][-1][:, 4097:4101]
    #
    #     final_output = tf.add(output_fw_pred, output_bw_pred) / 2  # shape:(batch_size,4)
    #     # outputs = tf.concat(output_bi_lstm,2) #(n_step,batchsize,num_input*2)
    #     # outputs[-1][:,4097:4101]
    #     return final_output  # last time_step output,(1,tensor([batch_size,num_input*2]))

    def lstm_single_2(self, x_input):
        x_in = tf.transpose(x_input, [1, 0, 2])  # [n_step, batch_size, num_input]
        lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.num_unit, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.num_unit, forget_bias=1.0, state_is_tuple=True)
        with tf.variable_scope('bidirectional_lstm'):
            # forward direction
            with tf.variable_scope('fw_direction') as fw_scope:
                outputs_fw, states_fw = tf.nn.dynamic_rnn(lstm_cell_fw, x_in, dtype=tf.float32, time_major=True)
            with tf.variable_scope('bw_directional'):
                input_reverse = tf.reverse(x_in,axis=[0])
                tmp, states_bw = tf.nn.dynamic_rnn(lstm_cell_bw,input_reverse,dtype=tf.float32, time_major=True)
                outputs_bw = tf.reverse(tmp, axis=[0])

        output_fw = tf.layers.dense(outputs_fw[-1], units=self.num_gt)  # limit output to num_gt via a fully connected layer
        output_bw = tf.layers.dense(outputs_bw[-1], units=self.num_gt)
        final_out = tf.add(output_fw, output_bw) /2
        return final_out

    def bi_lstm_2(self, name, X):
        x_in = tf.transpose(X, [1, 0, 2])  # [n_step,batch_size,num_input]
        with tf.variable_scope('forward'):
            lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(self.num_input)
        with tf.variable_scope('backward'):
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
        self.lstm_module = self.lstm_single_2(self.x)
        # self.lstm_module = self.bi_lstm_2('bi_lstm_2',self.x)
        self.ious = tf.Variable(tf.zeros([self.batch_size]), name="ious")
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        # self.saver.restore(self.sess, self.rolo_weights_file)
        if self.disp_console: print "Loading complete!" + '\n'

    def training(self, x_path, y_path):
        total_loss = 0

        if self.disp_console: print("TRAINING ROLO...")
        # Use rolo_input for LSTM training
        # pred = self.LSTM_single('lstm_train', self.x, self.istate, self.weights, self.biases)
        pred = self.lstm_single_2(self.x)
        if self.disp_console: print("pred: ", pred)
        self.pred_location = pred[0][:, 4097:4101]
        if self.disp_console: print("pred_location: ", self.pred_location)
        if self.disp_console: print("self.y: ", self.y)

        self.correct_prediction = tf.square(self.pred_location - self.y)
        if self.disp_console: print("self.correct_prediction: ", self.correct_prediction)
        self.accuracy = tf.reduce_mean(self.correct_prediction) * 100
        if self.disp_console: print("self.accuracy: ", self.accuracy)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.accuracy)  # Adam Optimizer

        # Initializing the variables
        init = tf.global_variables_initializer()

        # Launch the graph
        with tf.Session() as sess:

            if self.restore_weights == True:
                sess.run(init)
                self.saver.restore(sess, self.rolo_weights_file)
                print "Loading complete!" + '\n'
            else:
                sess.run(init)

            id = 0

            # Keep training until reach max iterations
            while id * self.batch_size < self.training_iters:
                # Load training data & ground truth
                batch_xs = self.rolo_utils.load_yolo_output(x_path, self.batch_size, self.num_steps,
                                                            id)  # [num_of_examples, num_input] (depth == 1)
                print('len(batch_xs)= ', len(batch_xs))
                # for item in range(len(batch_xs)):

                batch_ys = self.rolo_utils.load_rolo_gt(y_path, self.batch_size, self.num_steps, id)
                batch_ys = utils.locations_from_0_to_1(self.w_img, self.h_img, batch_ys)

                # Reshape data to get 3 seq of 5002 elements
                batch_xs = np.reshape(batch_xs, [self.batch_size, self.num_steps, self.num_input])
                batch_ys = np.reshape(batch_ys, [self.batch_size, 4])
                if self.disp_console: print("Batch_ys: ", batch_ys)

                pred_location = sess.run(self.pred_location, feed_dict={self.x: batch_xs, self.y: batch_ys,
                                                                        self.istate: np.zeros(
                                                                            (self.batch_size, 2 * self.num_input))})
                if self.disp_console: print("ROLO Pred: ", pred_location)
                # print("len(pred) = ", len(pred_location))
                if self.disp_console: print(
                "ROLO Pred in pixel: ", pred_location[0][0] * self.w_img, pred_location[0][1] * self.h_img,
                pred_location[0][2] * self.w_img, pred_location[0][3] * self.h_img)
                # print("correct_prediction int: ", (pred_location + 0.1).astype(int))

                # Save pred_location to file
                utils.save_rolo_output(self.output_path, pred_location, id, self.num_steps, self.batch_size)

                sess.run(optimizer, feed_dict={self.x: batch_xs, self.y: batch_ys,
                                               self.istate: np.zeros((self.batch_size, 2 * self.num_input))})
                if id % self.display_step == 0:
                    # Calculate batch loss
                    loss = sess.run(self.accuracy, feed_dict={self.x: batch_xs, self.y: batch_ys, self.istate: np.zeros(
                        (self.batch_size, 2 * self.num_input))})
                    if self.disp_console: print "Iter " + str(
                        id * self.batch_size) + ", Minibatch Loss= " + "{:.6f}".format(
                        loss)  # + "{:.5f}".format(self.accuracy)
                    total_loss += loss
                id += 1
                if self.disp_console: print(id)

                # show 3 kinds of locations, compare!

            print "Optimization Finished!"
            avg_loss = total_loss / id
            print "Avg loss: " + str(avg_loss)
            save_path = self.saver.save(sess, self.rolo_weights_file)
            print("Model saved in file: %s" % save_path)

        return avg_loss

    def train_20(self):
        print("TRAINING ROLO...")
        log_file = open("panchen/output/trainging-20-log.txt", "a")  # open in append mode
        print ("build_network...")
        self.build_networks()

        ''' TUNE THIS'''
        num_videos = 20
        epoches = 20 * 100  # 20 * 100

        # Use rolo_input for LSTM training
        with tf.variable_scope('opt'):
            # pred = self.LSTM_single('lstm_train', self.x, self.istate, self.weights, self.biases)
            # self.pred_location = pred[0][:, 4097:4101] #[batch_size,4097:4101]
            # self.pred_location = self.bi_lstm("bi_lstm_train", self.x)
            # self.pred_location = self.bi_lstm_2('bi_lstm_2',self.x)
            self.pred_location = self.lstm_single_2(self.x)
            self.correct_prediction = tf.square(self.pred_location - self.y)
            self.accuracy = tf.reduce_mean(self.correct_prediction) * 100
            self.learning_rate = 0.00001
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
                self.accuracy)  # Adam Optimizer

        # Initializing the variables
        init = tf.global_variables_initializer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # Launch the graph
        with tf.Session(config=config) as sess:
            if (self.restore_weights == True):
                sess.run(init)
                self.saver.restore(sess, self.rolo_weights_file)
                print "Loading complete!" + '\n'
            else:
                sess.run(init)

            for epoch in range(epoches):  # 20
                i = epoch % num_videos  # 20
                [self.w_img, self.h_img, sequence_name, dummy, self.training_iters] = utils.choose_video_sequence(i)

                x_path = os.path.join('../../benchmark/DATA', sequence_name, 'yolo_out/')
                y_path = os.path.join('../../benchmark/DATA', sequence_name, 'groundtruth_rect.txt')
                self.output_path = os.path.join('../../benchmark/DATA', sequence_name, 'rolo_out_train/')
                utils.createFolder(self.output_path)
                total_loss = 0
                id = 0
                total_time = 0
                start_time = time.time()

                # Keep training until reach max iterations
                while id < self.training_iters - self.num_steps:
                    # Load training data & ground truth
                    batch_xs = self.rolo_utils.load_yolo_output_test(x_path, self.batch_size, self.num_steps,
                                                                     id)  # [num_of_examples, num_input] (depth == 1)

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

                    pred_location = sess.run(self.pred_location, feed_dict={self.x: batch_xs, self.y: batch_ys,
                                                                            self.istate: np.zeros(
                                                                                (self.batch_size, 2 * self.num_input))})
                    if self.disp_console: print("ROLO Pred: ", pred_location)
                    # print("len(pred) = ", len(pred_location))
                    if self.disp_console: print(
                    "ROLO Pred in pixel: ", pred_location[0][0] * self.w_img, pred_location[0][1] * self.h_img,
                    pred_location[0][2] * self.w_img, pred_location[0][3] * self.h_img)
                    # print("correct_prediction int: ", (pred_location + 0.1).astype(int))

                    # Save pred_location to file
                    utils.save_rolo_output(self.output_path, pred_location, id, self.num_steps, self.batch_size)

                    sess.run(self.optimizer, feed_dict={self.x: batch_xs, self.y: batch_ys,
                                                        self.istate: np.zeros((self.batch_size, 2 * self.num_input))})

                    if id % self.display_step == 0:
                        # Calculate batch loss
                        loss = sess.run(self.accuracy, feed_dict={self.x: batch_xs, self.y: batch_ys,
                                                                  self.istate: np.zeros(
                                                                      (self.batch_size, 2 * self.num_input))})
                        if self.disp_console: print "Iter " + str(
                            id * self.batch_size) + ", Minibatch Loss= " + "{:.6f}".format(
                            loss)  # + "{:.5f}".format(self.accuracy)
                        total_loss += loss
                    id += 1
                    if self.disp_console: print(id)

                cycle_time = time.time() - start_time
                print('epoch is %d, time is %.2f' % (epoch, cycle_time))
                total_time += cycle_time
                if i + 1 == num_videos:
                    print 'total_time is %.2f' % total_time
                # print "Optimization Finished!"
                avg_loss = total_loss / id
                print "Avg loss: " + sequence_name + ": " + str(avg_loss)

                log_file.write(str("{:.3f}".format(avg_loss)) + '  ')
                if i + 1 == num_videos:
                    log_file.write('\n')
                    save_path = self.saver.save(sess, self.rolo_weights_file)
                    print("Model saved in file: %s" % save_path)

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
