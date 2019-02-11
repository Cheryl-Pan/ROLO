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
Script File: ROLO_step6_train_30_exp3.py

Description:

	ROLO is short for Recurrent YOLO, aimed at simultaneous object detection and tracking
	Paper: http://arxiv.org/abs/1607.05781
	Author: Guanghan Ning
	Webpage: http://guanghan.info/
'''

# Imports
import os, sys

abspath = os.path.abspath("..")  # ~/ROLO/experiments
rootpath = os.path.split(abspath)[0]  # ~/ROLO
sys.path.append(rootpath)
import utils.ROLO_utils as utils

import tensorflow as tf
# from tensorflow.models.rnn import rnn, rnn_cell
import cv2

import numpy as np
import os.path
import time
import random


class ROLO_TF:
    disp_console = False
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
    rolo_model_file = 'panchen/output/model_step4_exp2/'
    rolo_weights_file = os.path.join(rolo_model_file, 'model_step4_exp2.ckpt')
    lstm_depth = 3
    num_steps = 4  # number of frames as an input sequence
    num_feat = 4096
    num_predict = 6  # final output of LSTM 6 loc parameters
    num_gt = 4
    num_input = num_feat + num_predict  # data input: 4096+6= 5002

    # ROLO Training Parameters
    learning_rate = 0.00001

    training_iters = 210  # 100000
    batch_size = 1  # 128
    display_step = 1

    # tf Graph input
    x = tf.placeholder("float32", [None, num_steps, num_input])
    istate = tf.placeholder("float32", [None, 2 * num_input])  # state & cell => 2x num_input
    y = tf.placeholder("float32", [None, num_gt])

    # Define weights
    # weights = {
    #     'out': tf.Variable(tf.random_normal([num_input, num_predict]))
    # }
    # biases = {
    #     'out': tf.Variable(tf.random_normal([num_predict]))
    # }

    def __init__(self, argvs=[]):
        print("ROLO init")
        self.ROLO(argvs)

    def bi_lstm(self, name, X):
        _X = tf.reshape(X, [-1, self.num_input])
        x_in = tf.reshape(_X, [-1, self.num_steps, self.num_input])
        x_in = tf.transpose(x_in, [1, 0, 2])  # [n_step,batch_size,num_input]
        lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(self.num_input)
        lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(self.num_input)

        output_bi_lstm, states = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, x_in, dtype=tf.float32,
                                                                 time_major=True)
        # output_bi_lstm is a tuple,(output_fw,output_bw)
        output_fw_pred = output_bi_lstm[0][-1][:, 4097:4101]
        output_bw_pred = output_bi_lstm[1][-1][:, 4097:4101]
        out = tf.add(output_fw_pred, output_bw_pred, name="add") / 2

        return [output_fw_pred, output_bw_pred, out]

    # Experiment with dropout
    def dropout_features(self, feature, prob):
        if prob == 0:
            return feature
        else:
            num_drop = int(prob * 4096)
            drop_index = random.sample(xrange(4096), num_drop)
            for i in range(len(drop_index)):
                index = drop_index[i]
                feature[index] = 0
            return feature

        # Experiment with input box noise (translate, scale)

    def det_add_noise(self, det):
        translate_rate = random.uniform(0.98, 1.02)
        scale_rate = random.uniform(0.8, 1.2)

        det[0] *= translate_rate
        det[1] *= translate_rate
        det[2] *= scale_rate
        det[3] *= scale_rate

        return det

    '''---------------------------------------------------------------------------------------'''

    def build_networks(self):
        if self.disp_console: print "Building ROLO graph..."

        # Build rolo layers
        self.lstm_module = self.LSTM_single('lstm_test', self.x, self.istate, self.weights, self.biases)
        self.ious = tf.Variable(tf.zeros([self.batch_size]), name="ious")
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()
        # self.saver.restore(self.sess, self.rolo_weights_file)
        if self.disp_console: print "Loading complete!" + '\n'

    def train_30_2(self):
        print("TRAINING ROLO...")
        log_file = open("output/trainging-step6-30-2-log.txt", "a")  # open in append mode
        # self.build_networks()

        ''' TUNE THIS'''
        num_videos = 30
        epoches = 30 * 200

        # Use rolo_input for LSTM training
        [fw_prediction, bw_prediction, pred] = self.bi_lstm("bi_lstm", self.x)
        correct_prediction = tf.square(pred - self.y)
        lstm_pred_1 = tf.square(fw_prediction - self.y)
        lstm_pred_2 = tf.square(bw_prediction - self.y)
        accuracy = (0.5 * tf.reduce_mean(correct_prediction) + 0.25 * tf.reduce_mean(lstm_pred_1)
                    + 0.25 * tf.reduce_mean(lstm_pred_2)) * 100
        learning_rate = 0.00001
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(accuracy)  # Adam Optimizer

        # Initializing the variables
        init = tf.initialize_all_variables()
        self.saver = tf.train.Saver(max_to_keep=1)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # Launch the graph
        with tf.Session(config=config) as sess:
            sess.run(init)
            ckpt = tf.train.get_checkpoint_state(self.rolo_model_file)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(sess, ckpt.mdoel_checkpoint_path)
                print "Loading complete!" + '\n'

            for epoch in range(epoches):
                i = epoch % num_videos
                [self.w_img, self.h_img, sequence_name, self.training_iters, dummy] = utils.choose_video_sequence(i)

                x_path = os.path.join('benchmark/DATA', sequence_name, 'yolo_out/')
                y_path = os.path.join('benchmark/DATA', sequence_name, 'groundtruth_rect.txt')
                self.output_path = os.path.join('benchmark/DATA', sequence_name, 'rolo_out_train/')
                utils.createFolder(self.output_path)
                total_loss = 0
                id = 0

                # Keep training until reach max iterations
                while id < self.training_iters - self.num_steps:
                    # Load training data & ground truth
                    batch_xs = self.rolo_utils.load_yolo_output_test(x_path, self.batch_size, self.num_steps,
                                                                     id)  # [num_of_examples, num_input] (depth == 1)

                    # Apply dropout to batch_xs
                    # for item in range(len(batch_xs)):
                    #    batch_xs[item] = self.dropout_features(batch_xs[item], 0)

                    # print(id)
                    batch_ys = self.rolo_utils.load_rolo_gt_test(y_path, self.batch_size, self.num_steps, id)
                    batch_ys = utils.locations_from_0_to_1(self.w_img, self.h_img, batch_ys)

                    # Reshape data to get 3 seq of 5002 elements
                    batch_xs = np.reshape(batch_xs, [self.batch_size, self.num_steps, self.num_input])
                    batch_ys = np.reshape(batch_ys, [self.batch_size, 4])
                    if self.disp_console: print("Batch_ys: ", batch_ys)

                    pred_location = sess.run(pred, feed_dict={self.x: batch_xs, self.y: batch_ys, self.istate: np.zeros(
                        (self.batch_size, 2 * self.num_input))})
                    if self.disp_console: print("ROLO Pred: ", pred_location)
                    # print("len(pred) = ", len(pred_location))
                    if self.disp_console: print(
                    "ROLO Pred in pixel: ", pred_location[0][0] * self.w_img, pred_location[0][1] * self.h_img,
                    pred_location[0][2] * self.w_img, pred_location[0][3] * self.h_img)
                    # print("correct_prediction int: ", (pred_location + 0.1).astype(int))

                    # Save pred_location to file
                    utils.save_rolo_output_test(self.output_path, pred_location, id, self.num_steps, self.batch_size)

                    sess.run(self.optimizer, feed_dict={self.x: batch_xs, self.y: batch_ys,
                                                        self.istate: np.zeros((self.batch_size, 2 * self.num_input))})
                    if id % self.display_step == 0:
                        # Calculate batch loss
                        loss = sess.run(accuracy, feed_dict={self.x: batch_xs, self.y: batch_ys, self.istate: np.zeros(
                            (self.batch_size, 2 * self.num_input))})
                        if self.disp_console: print "Iter " + str(
                            id * self.batch_size) + ", Minibatch Loss= " + "{:.6f}".format(
                            loss)  # + "{:.5f}".format(self.accuracy)
                        total_loss += loss
                    id += 1
                    if self.disp_console: print(id)

                # print "Optimization Finished!"
                avg_loss = total_loss / id
                print "Avg loss: " + sequence_name + ": " + str(avg_loss)

                log_file.write(str("{:.3f}".format(avg_loss)) + '  ')
                if i + 1 == num_videos:
                    log_file.write('\n')
                    save_path = self.saver.save(sess, self.rolo_weights_file, global_step=epoch + 1)
                    print("Model saved in file: %s" % save_path)

        log_file.close()
        return

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
            self.train_30_2()

    '''----------------------------------------main-----------------------------------------------------'''


def main(argvs):
    ROLO_TF(argvs)


if __name__ == '__main__':
    main(' ')

