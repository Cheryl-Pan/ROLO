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
Script File: ROLO_network_test_all.py

Description:

	ROLO is short for Recurrent YOLO, aimed at simultaneous object detection and tracking
	Paper: http://arxiv.org/abs/1607.05781
	Author: Guanghan Ning
	Webpage: http://guanghan.info/
'''

# Imports
import os, sys

abspath = os.path.abspath("..")  # ~/ROLO/experiments
print abspath
rootpath = os.path.split(abspath)[0]  # ~/ROLO
sys.path.append(rootpath)
from utils import ROLO_utils as utils

import tensorflow as tf
# from tensorflow.models.rnn import rnn, rnn_cell
import cv2

import numpy as np
import os.path
import time
import random


class ROLO_TF:
    disp_console = True
    restore_weights = True  # False

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

    rolo_weights_file = '../training/panchen/output/ROLO_model'
    lstm_depth = 3
    num_steps = 6  # number of frames as an input sequence
    num_feat = 4096
    num_predict = 6  # final output of LSTM 6 loc parameters
    num_gt = 4
    num_input = num_feat + num_predict  # data input: 4096+6= 5002
    num_unit = num_input

    # ROLO Parameters
    batch_size = 1
    display_step = 1

    # tf Graph input
    x = tf.placeholder("float32", [None, num_steps, num_input], name='input_x')
    istate = tf.placeholder("float32", [None, 2 * num_input], name='y')  # state & cell => 2x num_input
    y = tf.placeholder("float32", [None, num_gt])

    # Define weights

    with tf.variable_scope("weight", reuse=True):
        weights = {
            'out': tf.Variable(tf.random_normal([num_input, num_gt]))
        }
    with tf.variable_scope("bias", reuse=True):
        biases = {
            'out': tf.Variable(tf.random_normal([num_gt]))
        }

    def __init__(self, argvs=[]):
        print("ROLO init")
        # tf.reset_default_graph()
        self.ROLO(argvs)

    def LSTM_single(self, name, _X, _istate, _weights, _biases):
        with tf.device('/gpu:0'):
            # input shape: (batch_size, n_steps, n_input)
            _X = tf.transpose(_X, [1, 0, 2])  # permute num_steps and batch_size
            # Reshape to prepare input to hidden activation
            _X = tf.reshape(_X, [self.num_steps * self.batch_size, self.num_input])  # (num_steps*batch_size, num_input)
            # Split data because rnn cell needs a list of inputs for the RNN inner loop
            _X = tf.split(0, self.num_steps, _X)  # n_steps * (batch_size, num_input)

        cell = tf.nn.rnn_cell.LSTMCell(self.num_input, self.num_input)
        state = _istate
        for step in range(self.num_steps):
            outputs, state = tf.nn.static_rnn(cell, [_X[step]], state)
            tf.get_variable_scope().reuse_variables()
        return outputs

    def bi_lstm(self, name, _X):
        _X = tf.transpose(_X, [1, 0, 2])  # [n_step,batch_size,num_input]
        with tf.variable_scope('forward'):
            lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(self.num_input)
        with tf.variable_scope('backward'):
            lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(self.num_input)

        output_bi_lstm, states = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, _X, dtype=tf.float32,
                                                                 time_major=True)
        # output_bi_lstm is a tuple,(output_fw,output_bw)
        output_fw_pred = output_bi_lstm[0][-1][:, 4097:4101]
        output_bw_pred = output_bi_lstm[1][-1][:, 4097:4101]

        final_output = tf.add(output_fw_pred, output_bw_pred) / 2  # shape:(batch_size,4)
        # outputs = tf.concat(output_bi_lstm,2) #(n_step,batchsize,num_input*2)
        # outputs[-1][:,4097:4101]
        return final_output  # last time_step output,(1,tensor([batch_size,num_input*2]))

    def lstm_single_2(self,name, x_input):
        x_in = tf.transpose(x_input, [1, 0, 2])  # [n_step, batch_size, num_input]
        lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.num_unit, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.num_unit, forget_bias=1.0, state_is_tuple=True)
        with tf.variable_scope('bidirectional_lstm')as scope:
            # forward direction
            with tf.variable_scope('fw_direction') as fw_scope:
                outputs_fw, states_fw = tf.nn.dynamic_rnn(lstm_cell_fw, x_in, dtype=tf.float32, time_major=True)
            with tf.variable_scope('bw_direction'):
                input_reverse = tf.reverse(x_in,axis=[0])
                tmp, states_bw = tf.nn.dynamic_rnn(lstm_cell_bw,input_reverse,dtype=tf.float32, time_major=True)
                outputs_bw = tf.reverse(tmp, axis=[0])

        out = tf.add(outputs_fw[-1], outputs_bw[-1])/2
        weight = self.weights['out']
        bias = self.biases['out']
        final_out = tf.matmul(out, weight) + bias
        return final_out

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
        self.ious = tf.Variable(tf.zeros([self.batch_size]), name="ious")
        # self.sess = tf.Session()
        # self.sess.run(tf.initialize_all_variables())
        # self.saver = tf.train.Saver()
        # self.saver.restore(self.sess, self.rolo_weights_file)
        # self.saver = tf.train.import_meta_graph("../training/panchen/output/ROLO_model/model_step6_exp1.ckpt.meta")

        if self.disp_console: print "Loading  graph complete!" + '\n'

    def testing(self):
        log_file = open("test-log.txt", 'a')
        # Use rolo_input for LSTM training
        # pred = self.LSTM_single('lstm_train', self.x, self.istate, self.weights, self.biases)
        # print("pred: ", pred)
        # self.pred_location = pred[0][:, 4097:4101]
        self.pred_location = self.lstm_single_2('bilstm',self.x)
        for v in tf.all_variables():
            print(v.name)
        # self.pred_location = tf.get_collection('lstm_single_2')
        self.correct_prediction = tf.square(self.pred_location - self.y)
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(self.correct_prediction) * 100
            tf.summary.scalar('loss', self.loss)

        merged_summary = tf.summary.merge_all()
        # Initializing the variables

        init = tf.global_variables_initializer()
        # include = ['bidirectional_lstm/bw_direction/rnn/basic_lstm_cell/kernel',
        #            'bidirectional_lstm/fw_direction/rnn/basic_lstm_cell/bias',
        #            'bidirectional_lstm/fw_direction/rnn/basic_lstm_cell/kernel',
        #            'bidirectional_lstm/bw_direction/rnn/basic_lstm_cell/bias',
        #            'weight/Variable'
        #            'bias/Variable']
        # variables_to_restore = tf.contrib.slim.get_variables_to_restore(include=include)
        # self.saver = tf.train.Saver(variables_to_restore)
        self.saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(self.rolo_weights_file)
        # self.saver = tf.train.import_meta_graph("../training/panchen/output/ROLO_model/model_step6_exp1.ckpt.meta")
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # Launch the graph
        with tf.Session(config=config) as sess:
            writer = tf.summary.FileWriter('log', sess.graph)
            if (self.restore_weights == True):
                sess.run(init)
                self.saver.restore(sess, ckpt)
                print ("Loading complete!" + '\n')
            else:
                sess.run(init)

            total_time = 0.0
            # id= 1
            evaluate_st = 22
            evaluate_ed = 22

            for test in range(evaluate_st, evaluate_ed + 1):
                [self.w_img, self.h_img, sequence_name, dummy_1, self.testing_iters] = utils.choose_video_sequence(test)

                x_path = os.path.join('../../benchmark/DATA', sequence_name, 'yolo_out/')
                y_path = os.path.join('../../benchmark/DATA', sequence_name, 'groundtruth_rect.txt')
                self.output_path = os.path.join('../../benchmark/DATA', sequence_name, 'rolo_out_test_fc/')
                utils.createFolder(self.output_path)
                print 'video: %d   TESTING ROLO on video sequence: %s' % (test + 1, sequence_name)
                # Keep training until reach max iterations
                total_loss = 0
                id = 0  # don't change this
                while id < self.testing_iters - self.num_steps:
                    # Load training data & ground truth
                    batch_xs = self.rolo_utils.load_yolo_output_test(x_path, self.batch_size, self.num_steps, id)  # [num_of_examples, num_input] (depth == 1)

                    # Apply dropout to batch_xs
                    # for item in range(len(batch_xs)):
                    #    batch_xs[item] = self.dropout_features(batch_xs[item], 0.4)

                    batch_ys = self.rolo_utils.load_rolo_gt_test(y_path, self.batch_size, self.num_steps, id)
                    batch_ys = utils.locations_from_0_to_1(self.w_img, self.h_img, batch_ys)

                    # Reshape data to get 3 seq of 5002 elements
                    batch_xs = np.reshape(batch_xs, [self.batch_size, self.num_steps, self.num_input])
                    batch_ys = np.reshape(batch_ys, [self.batch_size, 4])
                    # print("Batch_ys: ", batch_ys)

                    start_time = time.time()
                    pred_location = sess.run(self.pred_location, feed_dict={self.x: batch_xs})
                    cycle_time = time.time() - start_time
                    total_time += cycle_time
                    # print("ROLO Pred: ", pred_location)
                    # print("len(pred) = ", len(pred_location))
                    # print("ROLO Pred in pixel: ", pred_location[0][0]*self.w_img, pred_location[0][1]*self.h_img, pred_location[0][2]*self.w_img, pred_location[0][3]*self.h_img)
                    # print("correct_prediction int: ", (pred_location + 0.1).astype(int))

                    # Save pred_location to file
                    utils.save_rolo_output_test(self.output_path, pred_location, id, self.num_steps, self.batch_size)
                    if id % self.display_step == 0:
                        # Calculate batch loss
                        loss, summary = sess.run([self.loss, merged_summary],
                                                 feed_dict={self.x: batch_xs, self.y: batch_ys, self.istate: np.zeros(
                                                     (self.batch_size, 2 * self.num_input))})
                        writer.add_summary(summary, id)
                        # print "Iter " + str(id*self.batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) #+ "{:.5f}".format(self.accuracy)
                        total_loss += loss
                    id += 1

                avg_loss = total_loss / id
                print("Avg loss: " + str(avg_loss))
                print ("Time Spent on Tracking: " + str(total_time))
                print ("fps: " + str(id / total_time))
                print ("Testing Finished!")
                log_file.write(str("\nvideo: %d \nAvg loss: %.3f \nTime: %.3f \nfps: %.3f" % (
                test, avg_loss, total_time, id / total_time)))
        log_file.close()

        return None

    def ROLO(self, argvs):

        self.rolo_utils = utils.ROLO_utils()
        self.rolo_utils.loadCfg()
        self.params = self.rolo_utils.params

        arguments = self.rolo_utils.argv_parser(argvs)

        if self.rolo_utils.flag_train is True:
            self.training(utils.x_path, utils.y_path)
        elif self.rolo_utils.flag_track is True:
            self.build_networks()
            self.track_from_file(utils.file_in_path)
        elif self.rolo_utils.flag_detect is True:
            self.build_networks()
            self.detect_from_file(utils.file_in_path)
        else:
            print "Default: running ROLO test."
            # self.build_networks()

            self.testing()

    '''----------------------------------------main-----------------------------------------------------'''


def main(argvs):
    ROLO_TF(argvs)


if __name__ == '__main__':
    main(' ')

