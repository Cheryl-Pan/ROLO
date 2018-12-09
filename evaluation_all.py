import numpy
print numpy.__path__

import cv2
import os
import numpy as np
import sys
import utils.ROLO_utils as utils
import matplotlib.pyplot as plot
import pickle
import scipy.io
import re

def choose_benchmark_method(id):
    if id == 0:
        method = 'STRUCK'
    elif id == 1:
        method = 'CXT'
    elif id == 2:
        method = 'TLD'
    elif id == 3:
        method = 'OAB'
    elif id == 4:
        method = 'CSK'
    elif id == 5:
        method = 'RS'
    elif id == 6:
        method = 'LSK'
    elif id == 7:
        method = 'VTD'
    elif id == 8:
        method = 'VTS'
    elif id == 9:
        method = 'CNN-SVM'
    elif id == 10:
        method = 'Staple'
    return method

def choose_mat_file(method_id, sequence_id):
    [wid, ht, sequence_name, dummy_1, dummy_2] = utils.choose_video_sequence(sequence_id)
    method_name = choose_benchmark_method(method_id)
    mat_file = sequence_name + '_' + method_name + '.mat'

    return mat_file


def evaluate_AUC():        # calculate AUC(Average Under Curve)
    ''' PARAMETERS '''
    num_steps= 6

    evaluate_st = 0
    evaluate_ed = 29
    num_evaluate= evaluate_ed - evaluate_st + 1

    yolo_AUC_score= []
    rolo_AUC_score= []
    for thresh_int in range(0, 100, 5):
        thresh = thresh_int / 100.0  + 0.0001
        print("thresh= ", thresh)
        rolo_avg_score= 0
        yolo_avg_score= 0

        for test in range(evaluate_st, evaluate_ed + 1):

            [wid, ht, sequence_name, dummy_1, dummy_2] = utils.choose_video_sequence(test)

            img_fold_path = os.path.join('benchmark/DATA', sequence_name, 'img/')
            gt_file_path= os.path.join('benchmark/DATA', sequence_name, 'groundtruth_rect.txt')
            yolo_out_path= os.path.join('benchmark/DATA', sequence_name, 'yolo_out/')
            rolo_out_path= os.path.join('benchmark/DATA', sequence_name, 'rolo_out_test_all/')

            print(rolo_out_path)

            paths_imgs = utils.load_folder( img_fold_path)
            paths_rolo= utils.load_folder( rolo_out_path)
            lines = utils.load_dataset_gt( gt_file_path)

            # Define the codec and create VideoWriter object
            total= 0
            rolo_total_score= 0
            yolo_total_score= 0

            for i in range(len(paths_rolo)- num_steps):
                id= i + 1
                test_id= id + num_steps

                #path = paths_imgs[test_id]
                #img = utils.file_to_img(None, path)

                #if(img is None): break

                yolo_location= utils.find_yolo_location(yolo_out_path, test_id)
                yolo_location= utils.locations_normal(wid, ht, yolo_location)

                rolo_location= utils.find_rolo_location( rolo_out_path, test_id)
                rolo_location = utils.locations_normal( wid, ht, rolo_location)

                gt_location = utils.find_gt_location( lines, test_id - 1)

                rolo_score = utils.cal_rolo_score(rolo_location, gt_location, thresh)
                #print('rolo_score', rolo_score)
                rolo_total_score += rolo_score
                #print('rolo_total_score', rolo_total_score)
                yolo_score =  utils.cal_yolo_score(yolo_location, gt_location, thresh)
                yolo_total_score += yolo_score
                total += 1.0

            rolo_total_score /= total
            yolo_total_score /= total

            rolo_avg_score += rolo_total_score
            yolo_avg_score += yolo_total_score

            print('Sequence ID: ', test)
            print("yolo_avg_score = ", yolo_total_score)
            print("rolo_avg_score = ", rolo_total_score)

        yolo_AUC_score.append(yolo_avg_score/num_evaluate)
        rolo_AUC_score.append(rolo_avg_score/num_evaluate)

        print("(thresh, yolo_AUC_score) = ", thresh, ' ', yolo_avg_score/num_evaluate)
        print("(thresh, rolo_AUC_score) = ", thresh, ' ', rolo_avg_score/num_evaluate)

    with open('output/AUC_score_test_all.pickle', 'w') as f:
        pickle.dump([yolo_AUC_score, rolo_AUC_score], f)

    #draw_AUC()

def draw_AUC_OPE():

    num_methods = 9 + 1

    with open('output/AUC_score_test_all.pickle') as f:
        [yolo_AUC_score, rolo_AUC_score] = pickle.load(f)
    yolo_AUC_score.append(0)
    rolo_AUC_score.append(0)
    yolo_AUC_score = np.asarray(yolo_AUC_score)
    rolo_AUC_score = np.asarray(rolo_AUC_score)

    with open('output/evaluation/AUC_kalman_score.pickle') as f:
        [yolo_kalman_AUC_score] = pickle.load(f)
    yolo_kalman_AUC_score.append(0)
    yolo_kalman_AUC_score = np.asarray(yolo_kalman_AUC_score)


    benchmark_AUC_score = []
    for method_id in range(0, num_methods):
        method_name= choose_benchmark_method(method_id)
        file_name= 'output/evaluation/AUC_score_' + method_name + '.pickle'
        with open(file_name) as f:
            AUC_score = pickle.load(f)
            AUC_score.append(0)
            AUC_score = np.asarray(AUC_score)
            benchmark_AUC_score.append(AUC_score)


    x = [i/100.0 for i in range(0, 105, 5)]

    print(len(x))
    print(len(yolo_AUC_score))

    print(x)
    print(yolo_AUC_score)
    print(rolo_AUC_score)

    fig= plot.figure()
    ax = fig.gca()
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 100, 10))
    plot.title("Success Plot of OPE")
    #plot.title("Success Plot of OPE30: AUC(Average Under Curve)")
    plot.xlabel("overlap threshold")
    plot.ylabel("success rate")
    '''
    plot.plot(x, rolo_AUC_score*100, color = 'g', label = "ROLO", linestyle='-', marker= "s", markersize= 5, linewidth= 1, markevery= 1)
    plot.plot(x, yolo_AUC_score*100, color = 'g', label = "YOLO", linestyle='--', marker= "o", markersize= 5, linewidth= 1, markevery= 1)
    plot.plot(x, benchmark_AUC_score[0]*100, color = 'r', label = "STRUCK", linestyle='-', marker= "o", markersize= 5, linewidth= 1, markevery= 1)
    plot.plot(x, benchmark_AUC_score[1]*100, color = 'r', label = "CXT", linestyle='--', marker= "o", markersize= 5, linewidth= 1, markevery= 1)
    plot.plot(x, benchmark_AUC_score[2]*100, color = 'b', label = "TLD", linestyle='-', marker= "o", markersize= 5, linewidth= 1, markevery= 1)
    plot.plot(x, benchmark_AUC_score[3]*100, color = 'b', label = "OAB", linestyle='--', marker= "o", markersize= 5, linewidth= 1, markevery= 1)
    plot.plot(x, benchmark_AUC_score[4]*100, color = 'c', label = "CSK", linestyle='-', marker= "o", markersize= 5, linewidth= 1, markevery= 1)
    plot.plot(x, benchmark_AUC_score[5]*100, color = 'c', label = "RS", linestyle='--', marker= "o", markersize= 5, linewidth= 1, markevery= 1)
    plot.plot(x, benchmark_AUC_score[6]*100, color = 'm', label = "LSK", linestyle='-', marker= "o", markersize= 5, linewidth= 1, markevery= 1)
    plot.plot(x, benchmark_AUC_score[7]*100, color = 'm', label = "VTD", linestyle='--', marker= "o", markersize= 5, linewidth= 1, markevery= 1)
    plot.plot(x, benchmark_AUC_score[8]*100, color = 'y', label = "VTS", linestyle='-', marker= "o", markersize= 5, linewidth= 1, markevery= 1)
    '''

    'test all 30'
    #plot.plot(x, rolo_AUC_score*100, color = 'g', label = "ROLO [0.564]", linestyle='-',  markersize= 5, linewidth= 2, markevery= 1)   #exp all frames
    plot.plot(x, rolo_AUC_score*100, color = 'g', label = "ROLO [0.458]", linestyle='-',  markersize= 5, linewidth= 2, markevery= 1)  #exp 1/3 frames
    #plot.plot(x, benchmark_AUC_score[9]*100, color = 'y', label = "CNN-SVM[0.520]", linestyle='--', markersize= 5, linewidth= 2, markevery= 1)
    #plot.plot(x, yolo_AUC_score*100, color = 'g', label = "YOLO [0.440]", linestyle='--',  markersize= 5, linewidth= 2, markevery= 1)
    plot.plot(x, benchmark_AUC_score[0]*100, color = 'r', label = "STRUCK [0.410]", linestyle='-',  markersize= 5, linewidth= 2, markevery= 1)
    plot.plot(x, benchmark_AUC_score[3]*100, color = 'b', label = "OAB [0.366]", linestyle='--', markersize= 5, linewidth= 2, markevery= 1)
    plot.plot(x, benchmark_AUC_score[6]*100, color = 'm', label = "LSK [0.356]", linestyle='-',  markersize= 5, linewidth= 2, markevery= 1)
    plot.plot(x, benchmark_AUC_score[2]*100, color = 'b', label = "TLD [0.343]", linestyle='-',  markersize= 5, linewidth= 2, markevery= 1)

    plot.plot(x, yolo_kalman_AUC_score*100, color = 'k', label = "YOLO+SORT [0.341]", linestyle='--',  markersize= 5, linewidth= 2, markevery= 1)

    plot.plot(x, benchmark_AUC_score[1]*100, color = 'r', label = "CXT [0.333]", linestyle='--',  markersize= 5, linewidth= 2, markevery= 1)
    plot.plot(x, benchmark_AUC_score[5]*100, color = 'c', label = "RS [0.325]", linestyle='--',  markersize= 5, linewidth= 2, markevery= 1)
    plot.plot(x, benchmark_AUC_score[8]*100, color = 'y', label = "VTS [0.320]", linestyle='-',  markersize= 5, linewidth= 2, markevery= 1)
    plot.plot(x, benchmark_AUC_score[7]*100, color = 'm', label = "VTD [0.315]", linestyle='--',  markersize= 5, linewidth= 2, markevery= 1)
    plot.plot(x, benchmark_AUC_score[4]*100, color = 'c', label = "CSK [0.311]", linestyle='-', markersize= 5, linewidth= 2, markevery= 1)





    '''test last 8'''
    # plot.plot(x, rolo_AUC_score*100, color = 'g', label = "ROLO [0.476]", linestyle='-',  markersize= 5, linewidth= 2, markevery= 1)
    # plot.plot(x, yolo_AUC_score*100, color = 'g', label = "YOLO [0.459]", linestyle='--',  markersize= 5, linewidth= 2, markevery= 1)
    # plot.plot(x, benchmark_AUC_score[6]*100, color = 'm', label = "LSK [0.454]", linestyle='-',  markersize= 5, linewidth= 2, markevery= 1)
    # plot.plot(x, benchmark_AUC_score[8]*100, color = 'y', label = "VTS [0.444]", linestyle='-',  markersize= 5, linewidth= 2, markevery= 1)
    # plot.plot(x, benchmark_AUC_score[7]*100, color = 'm', label = "VTD [0.433]", linestyle='--',  markersize= 5, linewidth= 2, markevery= 1)
    # plot.plot(x, benchmark_AUC_score[1]*100, color = 'r', label = "CXT [0.433]", linestyle='--',  markersize= 5, linewidth= 2, markevery= 1)
    # plot.plot(x, benchmark_AUC_score[0]*100, color = 'r', label = "STRUCK [0.428]", linestyle='-',  markersize= 5, linewidth= 2, markevery= 1)
    # plot.plot(x, yolo_kalman_AUC_score*100, color = 'k', label = "YOLO+SORT [0.406]", linestyle='--',  markersize= 5, linewidth= 2, markevery= 1)
    # plot.plot(x, benchmark_AUC_score[4]*100, color = 'c', label = "CSK [0.406]", linestyle='-', markersize= 5, linewidth= 2, markevery= 1)
    # plot.plot(x, benchmark_AUC_score[5]*100, color = 'c', label = "RS [0.392]", linestyle='--',  markersize= 5, linewidth= 2, markevery= 1)
    # plot.plot(x, benchmark_AUC_score[3]*100, color = 'b', label = "OAB [0.366]", linestyle='--', markersize= 5, linewidth= 2, markevery= 1)
    # plot.plot(x, benchmark_AUC_score[2]*100, color = 'b', label = "TLD [0.318]", linestyle='-',  markersize= 5, linewidth= 2, markevery= 1)



    #plot.plot(x, benchmark_AUC_score[9]*100, color = 'y', label = "VTS", linestyle='--',  markersize= 5, linewidth= 2, markevery= 1)
    plot.axis([0, 1, 0, 100])

    plot.grid()
    plot.legend(loc = 1, prop={'size':10})
    plot.show()


if __name__=='__main__':
    evaluate_AUC()
    draw_AUC_OPE()