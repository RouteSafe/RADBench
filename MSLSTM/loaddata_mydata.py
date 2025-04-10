#_author_by_MC@20160424
import pywt
import matplotlib.pyplot as plt
import time
import random
import shutil
import os
import config
from collections import Counter
start = time.time()
import numpy as np
from numpy import *
from sklearn.preprocessing import LabelEncoder
from sklearn import svm,preprocessing
#from keras.utils import np_utils
import matplotlib
##
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import tensorflow as tf
import pandas as pd
flags = tf.app.flags
FLAGS = flags.FLAGS
import logging

# 配置日志 - 只输出到控制台
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def pprint(msg):
    logging.info(msg)
    #sys.stderr.write(msg+'\n')

def reverse_one_hot(y):
    """
    temp = []
    for tab1 in range(len(y)):
            temp.append(list(y[tab1]).index(max(list(y[tab1]))))
    return np.array(temp)"""
    if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
        return y
    else:
        return np.argmax(y, axis=1)
    #for i in range(len(y)):
        #for j in range(len(y[0])):
            #if y[i][j] == 1:
                #temp.append(j)
    #return np.array(temp)
def get_auc(arr_score, arr_label, pos_label):
    score_label_list = []
    for index in range(len(arr_score)):
        score_label_list.append((float(arr_score[index]), int(arr_label[index])))
    score_label_list_sorted = sorted(score_label_list, key = lambda line:line[0], reverse = True)

    fp, tp = 0, 0
    lastfp, lasttp = 0, 0
    A = 0
    lastscore = None

    for score_label in score_label_list_sorted:
        score, label = score_label[:2]
        if score != lastscore:
            A += trapezoid_area(fp, lastfp, tp, lasttp)
            lastscore = score
            lastfp, lasttp = fp, tp
        if label == pos_label:
            tp += 1
        else:
            fp += 1

    A += trapezoid_area(fp, lastfp, tp, lasttp)
    A /= (fp * tp)
    return A
def trapezoid_area(x1, x2, y1, y2):
    delta = abs(x2 - x1)
    return delta * 0.5 * (y1 + y2)
def get_data_path(filepath=None):
    if filepath is None or filepath == '':
        filepath = FLAGS.data_dir
    
    if not os.path.isabs(filepath):
        filepath = os.path.abspath(filepath)
    
    logging.info(f"使用数据路径: {filepath}")
    
    if not os.path.exists(filepath):
        logging.warning(f"警告: 目录不存在 {filepath}")
        try:
            os.makedirs(filepath, exist_ok=True)
            logging.info(f"创建数据目录: {filepath}")
        except Exception as e:
            logging.error(f"警告: 无法创建目录 {filepath}: {str(e)}")
    
    return filepath

def LoadData(input_data_path, filename):
    data_path = input_data_path
    
    file_path = os.path.join(data_path, filename)
    if not os.path.exists(file_path):
        logging.error(f"错误: 文件不存在: {file_path}")
        return None    
    with open(file_path) as fin:
        global negative_sign,positive_sign
        positive_sign = 1
        negative_sign = 0
        negative_flag = 0 # For binary txt flag is 1; for multi-class flag is 0
        Data=[]
        
        for each in fin:
            if '@' in each:continue
            val=each.split(",")
            if len(val)>0 or val[-1].strip()=="negative" or val[-1].strip()=="positive":
                if int(val[-1][0].strip()) == 1 or val[-1].strip()==negative_flag:#1 or 1.0 as the last element of the val
                #if val[-1].strip() != negative_flag:
                    val[-1] = negative_sign
                else:
                    val[-1] = positive_sign
                try:
                    val=list(map(lambda a:float(a),val))
                except:
                    val=list(map(lambda a:str(a),val))

                val[-1]=int(val[-1])
                Data.append(val)
        
        Data = np.array(Data)
        logging.info(f"Data: {Data.shape}")
        if len(Data) > 0:
            logging.info(f"特征数量: {Data.shape[1] - 1}")
        return Data

def process_single_file(fin, filename):
    global negative_sign, positive_sign
    
    data = []
    for line in fin:
        val = line.split(",")
        if len(val) > 0 or val[-1].strip() == "negative" or val[-1].strip() == "positive":
            if int(val[-1][0].strip()) == 1 or val[-1].strip() == 0:
                val[-1] = negative_sign
            else:
                val[-1] = positive_sign
                
            try:
                val = list(map(float, val))
            except:
                val = list(map(str, val))
                
            val[-1] = int(val[-1])
            data.append(val)
            
    return data

def Compute_average_list(mylist):
    temp = 0
    for i in range(len(mylist)):
        temp += float(mylist[i])
    return float(temp)/len(mylist)

def slidingFunc(window_size, data, label):
    """
    将数据转换为滑动窗口格式
    data: shape (samples, features)
    return: shape (n_windows, window_size, features)
    """
    newdata = []
    newlabel = []
    L = len(data)
    
    for index in range(0, L - window_size + 1):
        sequence = data[index:index + window_size]
        newdata.append(sequence)
        # 使用序列最后一个时间点的标签
        newlabel.append(label[index + window_size - 1])
        
    return np.array(newdata), np.array(newlabel)

def Manipulation(trainX,trainY,time_scale_size):
    window_size = len(trainX[0])
    temp = []
    N = window_size/time_scale_size
    for i in range(len(trainX)):
        temp.append([])
        for j in range(N):
            _value = np.zeros((1,len(trainX[0][0])))
            for k in range(time_scale_size):
                _value += trainX[i][j*time_scale_size+k]
            _value = _value/time_scale_size
            temp[i].extend(list(_value[0]))
    return temp,trainY
def returnPositiveIndex(data,negative_sign):
    temp = []
    for i in range(len(data)):
        if int(data[i]) != negative_sign:
            temp.append(i)
        else:
            continue
    return np.array(temp)

def returnNegativeIndex(data,negative_sign):
    temp = []
    for i in range(len(data)):
        if int(data[i]) == negative_sign:
            temp.append(i)
        else:
            continue
    return np.array(temp)
def returnSub_Positive(positive_list):
    temp1 = []
    temp2 = []
    flag = True
    for tab in range(len(positive_list)-1):
        if positive_list[tab+1] - positive_list[tab] < 2 and flag == True:
            temp1.append(positive_list[tab])
        else:
            flag = False
            temp2.append(positive_list[tab+1])
    return temp1,temp2

def Plotting_Sequence(X,Y):
    global output_folder
    for selected_feature in range(0,1):
        _index = [i for i in range(len(Y))]
        pos_index = returnPositiveIndex(Y,negative_sign=1)
        pos_index1,pos_index2 = returnSub_Positive(pos_index)

        plt.plot(_index,X[:,0][:,selected_feature],'b-',linewidth=2.0)
        plt.plot(pos_index1,X[pos_index1,0][:,selected_feature],'r-',linewidth=2.0)
        plt.plot(pos_index2,X[pos_index2,0][:,selected_feature],'r-',linewidth=2.0)
        plt.tick_params(labelsize=14)

        if selected_feature == 1:
            plt.ylim(-2,14)
        else:
            plt.ylim(-4,12)

        plt.savefig(os.path.join(os.path.join(os.getcwd(),output_folder),"F_"+str(selected_feature)\
                        +"_AAA"+str(random.randint(1,10000))+"AAAAAA.png"),dpi=100)

def MyEvaluation(Y_Testing, Result):
    acc = 0
    if len(Y_Testing)!=len(Result):
        print("Error!!!")
    else:
        for tab1 in range(len(Result)):
            temp_result = list(map(lambda a:int(round(a)),Result[tab1]))
            temp_true = list(map(lambda a:int(round(a)),Y_Testing[tab1]))
            if list(temp_result) == list(temp_true):
                acc += 1
    return round(float(acc)/len(Result),3)

def Multi_Scale_Wavelet(trainX,trainY,level,is_multi=True,wave_type='db1'):
    temp = [[] for i in range(level)]
    #temp = [[] for i in range(level + 1)]
    N = trainX.shape[0]
    if (is_multi == True) and (level > 1):
        for i in range(level):
        #for i in range(level+1):
            x = []
            for _feature in range(len(trainX[0])):
                coeffs = pywt.wavedec(trainX[:,_feature], wave_type, level=level)
                current_level = level  - i
                #current_level = level + 1 - i
                #for j in range(i+1,level+1):
                    #coeffs[j] = None
                #_rec = pywt.waverec(coeffs, wave_type)
                #x.append(_rec[:N])
                x.append(coeffs[i+1])

            temp[current_level - 1].extend(np.transpose(np.array(x)))

    else:
        for tab in range(level):
            current_level = level - tab
            temp[current_level - 1].extend(trainX)

    return  np.array(temp),trainX,trainY
def Multi_Scale_Wavelet0(trainX,trainY,level,is_multi=True,wave_type='db1'):
    temp = [[] for i in range(level)]
    N = trainX.shape[0]
    if (is_multi == True) and (level > 1):
        for i in range(level):
            x = []
            for _feature in range(len(trainX[0])):
                coeffs = pywt.wavedec(trainX[:,_feature], wave_type, level=level)
                current_level = level  - i
                for j in range(i+1,level+1):
                    coeffs[j] = None
                _rec = pywt.waverec(coeffs, wave_type)
                x.append(_rec[:N])

            temp[current_level - 1].extend(np.transpose(np.array(x)))

    else:
        for tab in range(level):
            current_level = level - tab
            temp[current_level - 1].extend(trainX)

    return  np.array(temp),trainX,trainY
def Multi_Scale_Wavelet000(trainX,trainY,level,is_multi=True,wave_type='db1'):
    temp = [[] for i in range(level)]
    x = np.transpose(trainX)
    if is_multi == True:
        for i in range(level):
            coeffs = pywt.wavedec(x[:,0], wave_type, level=level)
            current_level = level - i
            for j in range(i+1,level+1):
                coeffs[j] = None
            _rec = pywt.waverec(coeffs, wave_type)
            temp[current_level-1].extend(np.transpose(_rec))
    else:
        for tab in range(level):
            current_level = level - tab
            temp[current_level - 1].extend(np.transpose(x))

    return  np.array(temp),trainX,trainY
def return_indexes(index_,label):
    index_1 = []
    index_2 = []
    flag = False
    index_Anomaly = index_[label == 1]
    for tab_ in range(len(index_Anomaly) - 1):
        if index_Anomaly[tab_ + 1] - index_Anomaly[tab_] > 2:
            flag = True
        if flag == True:
            try:
                index_2.append(index_Anomaly[tab_ + 1])
            except:
                pass
        else:
            index_1.append(index_Anomaly[tab_])
    # index_2.append(index_Anomaly[-1])
    index_1 = np.array(index_1)
    index_2 = np.array(index_2)
    return index_1,index_2
def multi_scale_plotting(dataA, label):
    """绘制多尺度图"""
    if dataA is None or label is None:
        logging.error("错误: 输入数据为空")
        return
        
    try:
        selected_feature = 2
        original = dataA[:,selected_feature]
        
        # 绘制原始数据
        plt.figure()
        plt.plot([i for i in range(len(original))], original, 'b')
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Original", fontsize=12)
        plt.tick_params(labelsize=12)
        plt.grid(True)
        plt.savefig("Original.png", dpi=400)
        plt.close()
        
        # 绘制多尺度图
        fig = plt.figure(figsize=(10,5))
        label_ = np.array(label)
        
        # 绘制子图
        ax1 = fig.add_subplot(221)
        index_ = np.array([i for i in range(len(original))])
        index_1, index_2 = return_indexes(index_, label_)
        
        plt.plot([i for i in range(len(original))], original, 'b')
        plt.plot(index_1, original[index_1], 'r')
        plt.plot(index_2, original[index_2], 'r')
        
        plt.xlabel("(a)", fontsize=10)
        plt.ylabel("Number of announced prefix")
        ax1.grid(True)
        plt.tick_params(labelsize=11)
        
        level_ = 2
        ax1 = fig.add_subplot(222)
        coeffs = pywt.wavedec(original, 'db1', level=level_)
        #newCoeffs = [None,coeffs[1],None]
        #new_ = pywt.waverec(newCoeffs,'db1')
        index_b = [ i for i in range(len(coeffs[1]))]
        plt.plot(index_b,coeffs[1],'b')
        #plt.plot(index_b[label_==1],new_[label_==1],'r.')
        #plt.plot([i for i in range(len(coeffs[1]))],coeffs[1],'b')
        plt.xlabel("(b)",fontsize=10)
        #plt.ylabel("Detail coefficient of level "+ str(level_))
        plt.ylabel("Number of announced prefix")
        ax1.grid(True)
        plt.tick_params(labelsize=11)

        ax1 = fig.add_subplot(223)
        level_ = 2
        coeffs = pywt.wavedec(original, 'db1', level=level_)
        newCoeffs = [None,coeffs[1],None]
        new_ = pywt.waverec(newCoeffs,'db1')
        #print(len(index_))
        #print(len(new_))
        index_ = [i for i in range(len(new_))]
        plt.plot(index_,new_,'b')
        plt.plot(index_1,new_[index_1],'r')
        plt.plot(index_2,new_[index_2],'r')
        #plt.plot([i for i in range(len(coeffs[2]))],coeffs[2],'b')
        plt.xlabel("(c)",fontsize=10)
        #plt.ylabel("Reconstructed detail of level "+str(level_))
        plt.ylabel("Number of announced prefix")
        ax1.grid(True)
        plt.tick_params(labelsize=11)


        ax1 = fig.add_subplot(224)
        level_ = 5
        coeffs = pywt.wavedec(original, 'db1', level=level_)
        newCoeffs = [coeffs[0],None,None,None,None,None]
        new_ = pywt.waverec(newCoeffs,'db1')

        index_ = np.array([i for i in range(len(new_))])
        index_1, index_2 = return_indexes(index_,label_)

        plt.plot(index_,new_,'b')
        plt.plot(index_1,new_[index_1],'r')
        plt.plot(index_2,new_[index_2],'r')
        plt.xlabel("(d)",fontsize=10)
        #plt.ylabel("Reconstructed Approximation of level "+str(level_))
        plt.ylabel("Number of announced prefix")
        ax1.grid(True)
        plt.tick_params(labelsize=11)



        """
        ax1 = fig.add_subplot(224)
        level_ = 3
        coeffs = pywt.wavedec(original, 'db1', level=level_)
        newCoeffs = [None,coeffs[1],None,None]
        new_ = pywt.waverec(newCoeffs,'db1')
        plt.plot([i for i in range(len(new_))],new_,'b')
        #plt.plot([i for i in range(len(coeffs[3]))],coeffs[3],'b')
        plt.xlabel("(d)")
        plt.ylabel("Details of level "+str(level_))
        ax1.grid(True)
        """
        plt.tight_layout()

        plt.savefig("Wavelet Decomposition.pdf", dpi=400)
        plt.close()
        
    except Exception as e:
        logging.error(f"绘图时出错: {str(e)}")
        logging.debug(f"数据形状: dataA={dataA.shape if dataA is not None else None}, label={len(label) if label is not None else None}")
def multi_scale_plotting_Multi(dataMulti,dataA):

    selected_feature = 1
    original = dataA[:,selected_feature]
    original_level1 = dataMulti[0][:,selected_feature]

    level2_A = dataMulti[1][:,selected_feature]
    #Obtain Level_2_D
    coeffs = pywt.wavedec(original, 'db1', level=1)
    newCoeffs = [None,coeffs[1]]
    level2_D = pywt.waverec(newCoeffs,'db1')

    level3_A = dataMulti[2][:,selected_feature]
    #Obtain Level_3_D
    coeffs = pywt.wavedec(original, 'db1', level=2)
    newCoeffs = [None,coeffs[1],None]
    level3_D = pywt.waverec(newCoeffs,'db1')

    level4_A = dataMulti[3][:,selected_feature]
    #Obtain Level_4_D
    coeffs = pywt.wavedec(original, 'db1', level=3)
    newCoeffs = [None,coeffs[1],None,None]
    level4_D = pywt.waverec(newCoeffs,'db1')

    x = [i + 1 for i in range(len(original_level1))]
    #fig = plt.figure(figsize=(20,10),dpi=400)
    fig = plt.figure()
    ymax = 2
    ymin = -2

    #ax1 = fig.add_subplot(4
    # 21)
    ax1 = fig
    plt.plot(x,level4_D,'g')


    plt.xlabel("Time",fontsize=12)
    plt.ylabel("Average Unique AS path length",fontsize=12)
    plt.tick_params(labelsize=12)
    plt.ylim(ymin,ymax)
    plt.grid(True)
    plt.savefig("Level_4_D.png",dpi=800)
    plt.show()

    ax1 = fig.add_subplot(422)
    ax1.plot(x,original_level1,'b')
    plt.xlabel("(b)")
    plt.ylabel("Average AS path length")
    plt.ylim(ymin,ymax)
    ax1.grid(True)

    ax1 = fig.add_subplot(423)
    ax1.plot(x,level2_A,'b')
    plt.xlabel("(c)")
    plt.ylabel("Average AS path length")
    plt.ylim(ymin,ymax)
    ax1.grid(True)

    ax1 = fig.add_subplot(424)
    ax1.plot(x,level2_D,'b')
    plt.xlabel("(d)")
    plt.ylabel("Average AS path length")
    ax1.grid(True)

    ax1 = fig.add_subplot(425)
    ax1.plot(x,level3_A,'b')
    plt.xlabel("(e)")
    plt.ylabel("Average AS path length")
    plt.ylim(ymin,ymax)
    ax1.grid(True)

    ax1 = fig.add_subplot(426)
    ax1.plot(x,level3_D,'b')
    plt.xlabel("(f)")
    plt.ylabel("Average AS path length")
    ax1.grid(True)

    ax1 = fig.add_subplot(427)
    ax1.plot(x,level4_A,'b')
    plt.xlabel("(g)")
    plt.ylabel("Average AS path length")
    plt.ylim(ymin,ymax)
    ax1.grid(True)

    ax1 = fig.add_subplot(428)
    ax1.plot(x,level4_D,'b')
    plt.xlabel("(h)")
    plt.ylabel("Average AS path length")
    ax1.grid(True)
    plt.show()

def converge(trainX,trainY,time_scale_size=1):
    window_size = len(trainX[0])
    temp = []
    N = window_size/time_scale_size
    for i in range(len(trainX)):
        temp.append([])
        for j in range(N):
            _value = np.zeros((1,len(trainX[0][0])))
            for k in range(time_scale_size):
                _value += trainX[i][j*time_scale_size+k]
            _value = _value/time_scale_size
            temp[i].append(list(_value[0]))

    return  np.array(temp),trainY
def Fun(multiscaleSequences,case = 'max pooling'):
    temp = []
    if case == 'diag pooling':
        if not len(multiscaleSequences) == len(multiscaleSequences[0]):
            print("-------------------------------------error!")
        else:
            for tab in range(len(multiscaleSequences)):
                temp.append(list(multiscaleSequences[tab][tab]))
    else:
        scale = len(multiscaleSequences)
        sequence_window = len(multiscaleSequences[0])
        dimensions = len(multiscaleSequences[0][0])
        for i in range(sequence_window):
            l = []
            for tab_scale in range(scale):
                l.append(multiscaleSequences[tab_scale][i])
            l = np.array(l)
            temp_sample = []
            for j in range(dimensions):
                if case == 'max pooling':
                    temp_sample.append(np.max(l[:,j]))# max pooling
                elif case == 'mean pooling':
                    temp_sample.append(np.mean(l[:,j]))# max pooling
            temp.append(temp_sample)
    return temp
def add_nosie(ratio,data):
    w = 5
    x = data[:,:-1]
    _std = x.std(axis=0,ddof=0)
    N = int(ratio*len(data))
    noise = []
    for i in range(N):
        baseinstance_index = random.randint(0,len(data)-1)
        base_instance = data[baseinstance_index]
        noise.append([])
        for j in range(len(_std)):
            temp = random.uniform(_std[j]*-1,_std[j])
            noise[i].append(float(base_instance[j]+temp/w))
        noise[i].append(base_instance[-1])
    noise = np.array(noise)
    return np.concatenate((data,noise),axis=0)
def mix_multi_scale1(trainX_multi,trainY,pooling_type):
    scale = len(trainX_multi)
    length = len(trainX_multi[0])
    temp_trainX = []
    for tab_length in range(length):
        total_ = []
        for tab_scale in range(scale):
            a = trainX_multi[tab_scale][tab_length]
            total_.append(a)
        b = Fun(total_,pooling_type)
        temp_trainX.append(b)
    return np.array(temp_trainX),trainY
def returnData_withval(dataX,dataY,is_binary_class):
    global positive_sign,negative_sign
    start_ratio = 0.8
    if is_binary_class:
        positive_index = returnPositiveIndex(dataY, negative_sign)
        negative_index = returnNegativeIndex(dataY, negative_sign)

        pos_train_index = positive_index[0:int(start_ratio * len(positive_index))]
        pos_val_index = positive_index[int(start_ratio * len(positive_index)):int((start_ratio + 0.2) * len(positive_index))]
        neg_train_index = negative_index[0:int(start_ratio * len(negative_index))]
        neg_val_index = negative_index[int(start_ratio * len(negative_index)):int((start_ratio + 0.2) * len(negative_index))]

        train_index = np.append(neg_train_index, pos_train_index, axis=0)
        val_index = np.append(neg_val_index, pos_val_index, axis=0)
        train_index.sort()
        val_index.sort()
        train_index = list(map(lambda a: int(a), train_index))
        val_index = list(map(lambda a: int(a), val_index))
        train_dataX = dataX[train_index]
        val_dataX = dataX[val_index]
        train_dataY = dataY[train_index]
        val_dataY = dataY[val_index]

    else:
         negative_sign = 0
         negative_index = returnNegativeIndex(dataY, negative_sign)
         neg_train_index = negative_index[0:int(start_ratio * len(negative_index))]
         neg_val_index = negative_index[int(start_ratio * len(negative_index)):int((start_ratio + 0.1) * len(negative_index))]
         neg_test_index = negative_index[int((start_ratio + 0.1) * len(negative_index)):len(negative_index) - 1]

         for tab_ in range(1,5):
             negative_sign = tab_
             positive_index = returnNegativeIndex(dataY, negative_sign)

             pos_train_index = positive_index[0:int(start_ratio * len(positive_index))]
             pos_val_index = positive_index[int(start_ratio * len(positive_index)):int((start_ratio + 0.1) * len(positive_index))]
             pos_test_index = positive_index[int((start_ratio + 0.1) * len(positive_index)):len(positive_index) - 1]

             train_index = np.append(neg_train_index, pos_train_index, axis=0)
             train_index.sort()
             neg_train_index = train_index

             val_index = np.append(neg_val_index, pos_val_index, axis=0)
             val_index.sort()
             neg_val_index = val_index

             test_index = np.append(neg_test_index, pos_test_index, axis=0)
             test_index.sort()
             neg_test_index = test_index

             # min_number = min(len(train_dataX),len(test_dataX))
             pprint("The training size is shapeAAA:")
             pprint("The POSITIVE TRAIN is " + str(len(pos_train_index)))
             pprint("The NEGATIVE TRAIN is " + str(len(neg_train_index)))
             pprint("The POSITIVE TEST is " + str(len(pos_test_index)))
             pprint("The NEGATIVE TEST is " + str(len(neg_test_index)))

         train_index = list(map(lambda a: int(a), train_index))
         train_dataX = dataX[train_index]
         train_dataY = dataY[train_index]

         val_index = list(map(lambda a: int(a), val_index))
         val_dataX = dataX[val_index]
         val_dataY = dataY[val_index]

         test_index = list(map(lambda a: int(a), test_index))
         test_dataX = dataX[test_index]
         test_dataY = dataY[test_index]


    #min_number = min(len(train_dataX),len(test_dataX))
    #pprint("The training size is shape:")
    #pprint(train_dataY.shape)
    #pprint("The POSITIVE is "+str(len(pos_train_index)))
    #pprint("The NEGATIVE is "+str(len(neg_train_index)))
    #pprint("The validation size is shape:")
    #pprint(val_dataX.shape)
    #pprint("The testing size is shape:")
    #pprint(test_dataX.shape)
    #pprint("The POSITIVE is "+str(len(pos_test_index)))
    #pprint("The NEGATIVE is "+str(len(neg_test_index)))

    return train_dataX,train_dataY,val_dataX,val_dataY

def returnData_withoutval(dataX,dataY,is_binary_class):
    global positive_sign,negative_sign
    start_ratio = 1
    if is_binary_class:
        positive_index = returnPositiveIndex(dataY, negative_sign)
        negative_index = returnNegativeIndex(dataY, negative_sign)

        pos_train_index = positive_index[0:int(start_ratio * len(positive_index))]
        neg_train_index = negative_index[0:int(start_ratio * len(negative_index))]

        train_index = np.append(neg_train_index, pos_train_index, axis=0)
        train_index.sort()
        train_index = list(map(lambda a: int(a), train_index))
        train_dataX = dataX[train_index]
        train_dataY = dataY[train_index]

    else:
         negative_sign = 0
         negative_index = returnNegativeIndex(dataY, negative_sign)
         neg_train_index = negative_index[0:int(start_ratio * len(negative_index))]
         neg_val_index = negative_index[int(start_ratio * len(negative_index)):int((start_ratio + 0.1) * len(negative_index))]
         neg_test_index = negative_index[int((start_ratio + 0.1) * len(negative_index)):len(negative_index) - 1]

         for tab_ in range(1,5):
             negative_sign = tab_
             positive_index = returnNegativeIndex(dataY, negative_sign)

             pos_train_index = positive_index[0:int(start_ratio * len(positive_index))]
             pos_val_index = positive_index[int(start_ratio * len(positive_index)):int((start_ratio + 0.1) * len(positive_index))]
             pos_test_index = positive_index[int((start_ratio + 0.1) * len(positive_index)):len(positive_index) - 1]

             train_index = np.append(neg_train_index, pos_train_index, axis=0)
             train_index.sort()
             neg_train_index = train_index

             val_index = np.append(neg_val_index, pos_val_index, axis=0)
             val_index.sort()
             neg_val_index = val_index

             test_index = np.append(neg_test_index, pos_test_index, axis=0)
             test_index.sort()
             neg_test_index = test_index

             # min_number = min(len(train_dataX),len(test_dataX))
             pprint("The training size is shapeAAA:")
             pprint("The POSITIVE TRAIN is " + str(len(pos_train_index)))
             pprint("The NEGATIVE TRAIN is " + str(len(neg_train_index)))
             pprint("The POSITIVE TEST is " + str(len(pos_test_index)))
             pprint("The NEGATIVE TEST is " + str(len(neg_test_index)))

         train_index = list(map(lambda a: int(a), train_index))
         train_dataX = dataX[train_index]
         train_dataY = dataY[train_index]

         val_index = list(map(lambda a: int(a), val_index))
         val_dataX = dataX[val_index]
         val_dataY = dataY[val_index]

         test_index = list(map(lambda a: int(a), test_index))
         test_dataX = dataX[test_index]
         test_dataY = dataY[test_index]


    #min_number = min(len(train_dataX),len(test_dataX))
    #pprint("The training size is shape:")
    #pprint(train_dataY.shape)
    #pprint("The POSITIVE is "+str(len(pos_train_index)))
    #pprint("The NEGATIVE is "+str(len(neg_train_index)))
    #pprint("The validation size is shape:")
    #pprint(val_dataX.shape)
    #pprint("The testing size is shape:")
    #pprint(test_dataX.shape)
    #pprint("The POSITIVE is "+str(len(pos_test_index)))
    #pprint("The NEGATIVE is "+str(len(neg_test_index)))
    return train_dataX,train_dataY

def return_tabData(current_cv=1, cross_cv=5, dataX=[], dataY=[]):
    global positive_sign, negative_sign
    
    # 确保dataX和dataY是numpy数组
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    
    # 获取正负样本索引，并确保返回的是整数类型
    positive_index = np.array(returnPositiveIndex(dataY, negative_sign), dtype=np.int32)
    negative_index = np.array(returnNegativeIndex(dataY, negative_sign), dtype=np.int32)
    
    logging.info(f"positive_index shape: {len(positive_index)}")
    logging.info(f"negative_index shape: {len(negative_index)}")
    logging.info(f"positive_index dtype: {positive_index.dtype}")
    logging.info(f"negative_index dtype: {negative_index.dtype}")

    # 使用整数类型的索引获取数据
    pos_data = dataX[positive_index]
    neg_data = dataX[negative_index]
    
    # 获取对应的标签
    pos_label = dataY[positive_index]
    neg_label = dataY[negative_index]
    
    for tab_cross in range(cross_cv):
        if not tab_cross == current_cv: continue
        pos_train_index = []
        pos_test_index = []
        neg_train_index = []
        neg_test_index = []

        for tab_positive in range(len(positive_index)):
            if int((cross_cv - tab_cross - 1) * len(pos_data) / cross_cv) <= tab_positive < int(
                                    (cross_cv - tab_cross) * len(pos_data) / cross_cv):
                pos_test_index.append(positive_index[tab_positive])
            else:
                pos_train_index.append(positive_index[tab_positive])

        for tab_negative in range(len(negative_index)):
            if int((cross_cv - tab_cross - 1) * len(neg_data) / cross_cv) <= tab_negative < int(\
                                    (cross_cv - tab_cross) * len(neg_data) / cross_cv):
                neg_test_index.append(negative_index[tab_negative])
            else:
                neg_train_index.append(negative_index[tab_negative])


        train_index = np.append(neg_train_index, pos_train_index, axis=0)
        train_index.sort()

        train_index = list(map(lambda a: int(a), train_index))
        #print(train_index)
        train_dataX = dataX[train_index, :]
        train_dataY = dataY[train_index]

        test_index = np.append(neg_test_index, pos_test_index, axis=0)
        test_index.sort()

        test_index = list(map(lambda a: int(a), test_index))
        test_dataX = dataX[test_index, :]
        test_dataY = dataY[test_index]

        #min_number = min(len(train_dataX),len(test_dataX))

        #pprint(str(tab_cross + 1) + "th Cross Validation of is running and the training size is shape("+str(len(train_dataX))+','+str(len(train_dataX[0]))+').')
        #pprint(str(tab_cross + 1) + "th Cross Validation of is running and the testing size is shape("+str(len(test_dataX))+','+str(len(test_dataX[0]))+').')
        #pprint(str(tab_cross + 1) + "th Cross Validation of is running and the testing size is shape("+str(len(test_dataY))+','+str(len(test_dataY[0]))+').')

        return train_dataX,train_dataY,test_dataX,test_dataY


def one_hot(y_):
    # Function to encode output labels from number indexes
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]

    y_ = y_.reshape(len(y_))
    n_values = int(np.max(y_) + 1)
    indexes = np.array(y_, dtype=np.int32)
    return np.eye(n_values)[indexes]  # Returns FLOATS
def get_testData(poolingType,isNoise,noiseRatio,filePath,fileName,windowSize,trigger_flag,multiScale=True,waveScale=-1,waveType="db1",timeScale=1):
    global positive_sign,negative_sign,output_folder,filename
    positive_sign = 1
    negative_sign = 0
    output_folder = "output"

    print(f"get_testData尝试从以下位置读取数据:{filePath}")
    data_test=LoadData(filePath,fileName)

    scaler = preprocessing.StandardScaler()

    #if multiClass=="Multi":np.random.shuffle(PositiveIndex)
    if multiScale == False:
        testX,testY = slidingFunc(windowSize, scaler.fit_transform(data_test[:, :-1]), data_test[:, -1])

    else:
        testX_Multi = [[] for i in range(waveScale)]

        print("wave type is "+ str(waveType)+" and the scale is "+str(waveScale))
        data_testMulti, data_testX, data_testY = Multi_Scale_Wavelet0(data_test[:, :-1],data_test[:, -1],waveScale,True,waveType)

        #Plot the different scale
        #multi_scale_plotting(Data_Multi_Level_X,Data_X)
        for tab_level in range(waveScale):
            data_testX_level,data_testY_level = slidingFunc(windowSize, scaler.fit_transform(data_testMulti[tab_level]), data_testY)
            testX_Multi[tab_level].extend(data_testX_level)
            testY = data_testY_level

    if trigger_flag:
        testY = one_hot(testY)

    if multiScale == False:
        return testX, testY

    testX_Multi = np.array(testX_Multi).transpose((1,0,2,3))#batch_size, scale_levels, sequence_window, input_dim
    print("Input shape is"+str(testX_Multi.shape))
    return testX_Multi,testY

def get_trainData(poolingType,isNoise,noiseRatio,filePath,fileName,windowSize,trigger_flag,multiScale=True,waveScale=-1,waveType="db1",timeScale=1):

    global positive_sign,negative_sign,output_folder
    positive_sign = 1
    negative_sign = 0
    output_folder = "output"

    #for tab in range(len(fileName)):
        #if tab == 0:
            #data_train_val=LoadData(filePath,fileNameList[0])
        #else:
            #temp = LoadData(filePath, fileNameList[tab])
            #data_train_val=np.concatenate((data_train_val,temp),axis=0)
    print(f"get_trainData尝试从以下位置读取数据:{filePath}")
    data_train_val=LoadData(filePath,fileName)
    scaler = preprocessing.StandardScaler()

    #Plotting_Sequence(Data_[:,0], Data_[:,-1])
    if isNoise == True:
        data_train_val = add_nosie(noiseRatio,data_train_val)

    #if multiClass=="Multi":np.random.shuffle(PositiveIndex)
    if multiScale == False:
        dataSequenlized_X,dataSequenlized_Y = slidingFunc(windowSize, scaler.fit_transform(data_train_val[:, :-1]), data_train_val[:, -1])
        trainX, trainY, valX, valY = return_tabData(dataX= dataSequenlized_X,dataY= dataSequenlized_Y)

    else:
        trainX_Multi = [[] for i in range(waveScale)]
        valX_Multi = [[] for i in range(waveScale)]

        print("wave type is "+ str(waveType)+" and the scale is "+str(waveScale))

        dataMulti, dataX, dataY = Multi_Scale_Wavelet0(data_train_val[:, :-1],data_train_val[:, -1],waveScale,True,waveType)

        #Plot the different scale
        #multi_scale_plotting(Data_Multi_Level_X,Data_X)
        for tab_level in range(waveScale):
            dataX_level,dataY_level = slidingFunc(windowSize, scaler.fit_transform(dataMulti[tab_level]), dataY)

            trainX,trainY,valX, valY = return_tabData(dataX= dataX_level,dataY= dataY_level)
            print("returnTabData_"+str(dataX_level.shape))
            trainX_Multi[tab_level].extend(trainX)
            valX_Multi[tab_level].extend(valX)

    if trigger_flag:
        trainY = one_hot(trainY)
        valY = one_hot(valY)

    if multiScale == False:
        return trainX, trainY, valX, valY

    trainX_Multi = np.array(trainX_Multi).transpose((1,0,2,3))#batch_size, scale_levels, sequence_window, input_dim
    valX_Multi = np.array(valX_Multi).transpose((1,0,2,3)) #batch_size, scale_levels, sequence_window, input_dim
    print("Input shape is"+str(trainX_Multi.shape))
    return trainX_Multi,trainY,valX_Multi,valY

def get_data(poolingType,isNoise,noiseRatio,filePath,fileName,windowSize,trigger_flag,is_binary_class,multiScale=True,waveScale=-1,waveType="db1",timeScale=1):
    # 确保使用绝对路径
    if not os.path.isabs(filePath):
        filePath = os.path.abspath(filePath)
            
    # 检查文件是否存在
    full_path = os.path.join(filePath, fileName)
    if not os.path.exists(full_path):
        print(f"错误: 文件不存在: {full_path}")
        return None, None
            
    data_train_val=LoadData(filePath,fileName)
    scaler = preprocessing.StandardScaler()

    #Plotting_Sequence(Data_[:,0], Data_[:,-1])
    if isNoise == True:
        data_train_val = add_nosie(noiseRatio,data_train_val)

    #if multiClass=="Multi":np.random.shuffle(PositiveIndex)
    if multiScale == False:
        dataSequenlized_X,dataSequenlized_Y = slidingFunc(windowSize, scaler.fit_transform(data_train_val[:, :-1]), data_train_val[:, -1])
        trainX, trainY, valX, valY = return_tabData(dataX= dataSequenlized_X,dataY= dataSequenlized_Y)

    else:
        trainX_Multi = [[] for i in range(waveScale)]
        valX_Multi = [[] for i in range(waveScale)]

        print("wave type is "+ str(waveType)+" and the scale is "+str(waveScale))

        dataMulti, dataX, dataY = Multi_Scale_Wavelet0(data_train_val[:, :-1],data_train_val[:, -1],waveScale,True,waveType)

        #Plot the different scale
        #multi_scale_plotting(Data_Multi_Level_X,Data_X)
        for tab_level in range(waveScale):
            dataX_level,dataY_level = slidingFunc(windowSize, scaler.fit_transform(dataMulti[tab_level]), dataY)

            trainX,trainY,valX, valY = return_tabData(dataX= dataX_level,dataY= dataY_level)
            print("returnTabData_"+str(dataX_level.shape))
            trainX_Multi[tab_level].extend(trainX)
            valX_Multi[tab_level].extend(valX)

    if trigger_flag:
        trainY = one_hot(trainY)
        valY = one_hot(valY)

    if multiScale == False:
        return trainX, trainY, valX, valY

    trainX_Multi = np.array(trainX_Multi).transpose((1,0,2,3))#batch_size, scale_levels, sequence_window, input_dim
    valX_Multi = np.array(valX_Multi).transpose((1,0,2,3)) #batch_size, scale_levels, sequence_window, input_dim
    print("Input shape is"+str(trainX_Multi.shape))
    return trainX_Multi,trainY,valX_Multi,valY

def get_data_withoutval(poolingType,isNoise,noiseRatio,filePath,fileName,windowSize,trigger_flag,is_binary_class,multiScale=True,waveScale=-1,waveType="db1",timeScale=1):
    # 确保使用绝对路径
    if not os.path.isabs(filePath):
        filePath = os.path.abspath(filePath)
            
    # 检查文件是否存在
    full_path = os.path.join(filePath, fileName)
    if not os.path.exists(full_path):
        print(f"错误: 文件不存在: {full_path}")
        return None, None
            
    data_train_val=LoadData(filePath,fileName)
    scaler = preprocessing.StandardScaler()

    #Plotting_Sequence(Data_[:,0], Data_[:,-1])
    if isNoise == True:
        data_train_val = add_nosie(noiseRatio,data_train_val)

    #if multiClass=="Multi":np.random.shuffle(PositiveIndex)
    if multiScale == False:
        dataSequenlized_X,dataSequenlized_Y = slidingFunc(windowSize, scaler.fit_transform(data_train_val[:, :-1]), data_train_val[:, -1])
        trainX, trainY = dataSequenlized_X, dataSequenlized_Y

    else:
        trainX_Multi = [[] for i in range(waveScale)]
        valX_Multi = [[] for i in range(waveScale)]

        print("wave type is "+ str(waveType)+" and the scale is "+str(waveScale))

        dataMulti, dataX, dataY = Multi_Scale_Wavelet0(data_train_val[:, :-1],data_train_val[:, -1],waveScale,True,waveType)

        #Plot the different scale
        #multi_scale_plotting(Data_Multi_Level_X,Data_X)
        for tab_level in range(waveScale):
            dataX_level,dataY_level = slidingFunc(windowSize, scaler.fit_transform(dataMulti[tab_level]), dataY)
            logging.info(f"dataX_level shape: {dataX_level.shape}")
            logging.info(f"dataY_level shape: {dataY_level.shape}")

            trainX,trainY,valX, valY = return_tabData(dataX= dataX_level,dataY= dataY_level)
            print("returnTabData_"+str(dataX_level.shape))
            trainX_Multi[tab_level].extend(trainX)
            valX_Multi[tab_level].extend(valX)
            valY = one_hot(valY)

    if trigger_flag:
        trainY = one_hot(trainY)

    if multiScale == False:
        return trainX, trainY

    trainX_Multi = np.array(trainX_Multi).transpose((1,0,2,3))#batch_size, scale_levels, sequence_window, input_dim
    valX_Multi = np.array(valX_Multi).transpose((1,0,2,3)) #batch_size, scale_levels, sequence_window, input_dim
    print("Input shape is"+str(trainX_Multi.shape))
    return trainX_Multi,trainY




def get_data_withoutS(poolingType, isNoise, noiseRatio, filePath, fileName, windowSize, trigger_flag, is_binary_class, multiScale=False,waveScale=-1,waveType="db1",timeScale=1):
    """加载并预处理数据"""
    try:
        # 加载原始数据
        data_ = LoadData(filePath, fileName)
        scaler = preprocessing.StandardScaler()
        if multiScale == False:
            trainX, trainY = returnData_withoutval(data_[:, :-1],data_[:, -1],is_binary_class)
            return trainX, trainY
        
    except Exception as e:
        logging.error(f"处理数据时出错: {str(e)}")
        return None, None

def GetData_WithoutS(Is_Adding_Noise,Noise_Ratio,Fila_Path,FileName,Window_Size,Current_CV,Cross_CV,Bin_or_Multi_Label="Bin",Multi_Scale=False,Wave_Let_Scale=2,Time_Scale_Size=1,Normalize=0):
    print("")
    global positive_sign,negative_sign
    output_folder = "output"

    if not os.path.isdir(os.path.join(os.getcwd(),output_folder)):
        os.makedirs(os.path.join(os.getcwd(),output_folder))
    positive_sign=1
    negative_sign=0
    print(f"GetData_WithoutS尝试从以下位置加载数据:{Fila_Path}")
    Data_=LoadData(Fila_Path,FileName)


    if Is_Adding_Noise == True:
        Data_ = add_nosie(Noise_Ratio,Data_)
    if Normalize == 1 or  Normalize==10 or Normalize==11:
        scaler = preprocessing.MinMaxScaler()
    elif Normalize == 2:
        scaler = preprocessing.Normalizer()
    else:
        scaler = preprocessing.StandardScaler()
    PositiveIndex = returnPositiveIndex(Data_,negative_sign)
    NegativeIndex = returnNegativeIndex(Data_,negative_sign)

    if Bin_or_Multi_Label=="Multi":np.random.shuffle(PositiveIndex)
    Pos_Data = Data_[PositiveIndex]
    Neg_Data = Data_[NegativeIndex]

    for tab_cross in range(Cross_CV):
        if not tab_cross == Current_CV: continue

        Positive_Data_Index_Training=[]
        Positive_Data_Index_Testing=[]
        Negative_Data_Index_Training=[]
        Negative_Data_Index_Testing=[]

        for tab_positive in range(len(PositiveIndex)):
            if int((Cross_CV-tab_cross-1)*len(Pos_Data)/Cross_CV)<=tab_positive<int((Cross_CV-tab_cross)*len(Pos_Data)/Cross_CV):
                Positive_Data_Index_Testing.append(PositiveIndex[tab_positive])
            else:
                Positive_Data_Index_Training.append(PositiveIndex[tab_positive])

        for tab_negative in range(len(NegativeIndex)):
            if int((Cross_CV-tab_cross-1)*len(Neg_Data)/Cross_CV)<=tab_negative<int((Cross_CV-tab_cross)*len(Neg_Data)/Cross_CV):
                Negative_Data_Index_Testing.append(NegativeIndex[tab_negative])
            else:
                Negative_Data_Index_Training.append(NegativeIndex[tab_negative])


        Training_Data_Index=np.append(Negative_Data_Index_Training,Positive_Data_Index_Training,axis=0)
        Training_Data_Index.sort()
        Training_Data_Index = list(map(lambda a:int(a),Training_Data_Index))
        Training_Data = Data_[Training_Data_Index,:]
        Testing_Data_Index=np.append(Negative_Data_Index_Testing,Positive_Data_Index_Testing,axis=0)
        Testing_Data_Index.sort()
        Testing_Data_Index = list(map(lambda a:int(a),Testing_Data_Index))
        Testing_Data = Data_[Testing_Data_Index,:]


        print(str(tab_cross + 1) + "th Cross Validation>>>>>>>> is running and the training size is " + str(
            len(Training_Data)) + ", testing size is " + str(len(Testing_Data)) + "......"+str(Normalize))


        #X_Training = np.array(scaler.fit_transform(Training_Data[:, :-1]))
        #Y_Training = np.array(Training_Data[:, -1])

        #X_Testing = np.array(scaler.fit_transform(Testing_Data[:, :-1]))
        #Y_Testing = np.array(Testing_Data[:, -1])

        #Y_Training_Encoder = LabelEncoder()
        #Y_Training_Encoder.fit(Y_Training)
        #encoded_Y1 = Y_Training_Encoder.transform(Y_Training)
        # convert integers to dummy variables (i.e. one hot encoded)
        #Y_Training = np_utils.to_categorical(encoded_Y1)
        Y_Training = one_hot(Y_Training)
        #Y_Testing_Encoder = LabelEncoder()
        #Y_Testing_Encoder.fit(Y_Testing)
        #encoded_Y2 = Y_Testing_Encoder.transform(Y_Testing)
        # convert integers to dummy variables (i.e. one hot encoded)
        #Y_Testing = np_utils.to_categorical(encoded_Y2)
        Y_Testing = one_hot(Y_Testing)

        return X_Training[20:],Y_Training[20:],np.array(Training_Data[:, -1])[20:],X_Testing[20:],Y_Testing[20:],np.array(Testing_Data[:, -1])[20:]


def get_all_subfactors(number):
    temp_list = []
    temp_list.append(1)
    temp_list.append(2)
    for i in range(3,number):
        if number%i == 0 :
            temp_list.append(i)
    temp_list.append(number)
    return temp_list
def split_data(direc,ratio,trainX,trainY,testX,testY):
  """Input:
  direc: location of the UCR archive
  ratio: ratio to split training and testset
  dataset: name of the dataset in the UCR archive"""
  print(trainX.shape)
  print(trainY.shape)


  DATA_X = np.concatenate((trainX,testX),axis=0)
  DATA_Y = np.concatenate((trainY,testY),axis=0)

  N = DATA_X.shape[0]

  ratio = (ratio*N).astype(np.int32)
  ind = np.random.permutation(N)
  print("NNNNNNNN")
  print(ind)
  print(ratio)
  X_train = DATA_X[ind[:ratio[0]],:]
  X_val = DATA_X[ind[ratio[0]:ratio[1]],:]
  X_test = DATA_X[ind[ratio[1]:],:]
  # Targets have labels 1-indexed. We subtract one for 0-indexed
  y_train = DATA_Y[ind[:ratio[0]]]
  y_val = DATA_Y[ind[ratio[0]:ratio[1]]]
  y_test = DATA_Y[ind[ratio[1]:]]
  return X_train,X_val,X_test,y_train,y_val,y_test

def set_style():
    plt.style.use(['seaborn-dark', 'seaborn-paper'])
    matplotlib.rc("font", family="serif")


if __name__=='__main__':
    global filepath, filename, fixed_seed_num, sequence_window, number_class, hidden_units, input_dim, learning_rate, epoch, is_multi_scale, training_level, cross_cv
    # ---------------------------Fixed Parameters------------------------------
    filepath = FLAGS.data_dir  # 使用 FLAGS 定义的路径
    logging.info(f"使用数据目录: {filepath}")
    
    # 确保使用绝对路径
    if not os.path.isabs(filepath):
        filepath = os.path.abspath(filepath)
    
    # 检查目录是否存在
    if not os.path.exists(filepath):
        logging.warning(f"警告: 目录不存在 {filepath}")
        try:
            os.makedirs(filepath, exist_ok=True)
            logging.info(f"创建数据目录: {filepath}")
        except Exception as e:
            logging.error(f"警告: 无法创建目录 {filepath}: {str(e)}")
    
    # 使用 FLAGS 定义的参数
    sequence_window = FLAGS.sequence_window
    hidden_units = FLAGS.num_neurons1
    input_dim = FLAGS.input_dim
    number_class = FLAGS.number_class
    cross_cv = 2  # 交叉验证折数
    fixed_seed_num = 1337  # 随机种子
    
    # 从数据目录中获取文件列表
    try:
        from pathlib import Path
        feature_dir = Path(filepath)
        filename_list = [f.name for f in feature_dir.glob('*.txt')]
        
        if not filename_list:
            logging.warning(f"警告: 在 {filepath} 中没有找到任何 .txt 文件")
            filename_list = ['HB_AS_Leak.txt']  # 默认文件名
        
        logging.info(f"在目录 {filepath} 中找到以下文件:")
        for f in filename_list:
            logging.info(f"- {f}")
    except Exception as e:
        logging.error(f"获取文件列表时出错: {str(e)}")
        filename_list = ['HB_AS_Leak.txt']  # 默认文件名
    
    filename = filename_list[0]  # 使用第一个文件
    
    # 其他参数
    learning_rate = FLAGS.learning_rate
    epoch = FLAGS.max_epochs
    is_add_noise = FLAGS.is_add_noise
    noise_ratio = FLAGS.noise_ratio
    wave_type = FLAGS.wave_type
    pooling_type = FLAGS.pooling_type
    is_multi_scale = FLAGS.is_multi_scale
    training_level = FLAGS.scale_levels
    
    for eachfile in filename_list:
        if '.py' in eachfile or '.DS_' in eachfile: 
            continue
        if not '.txt' in eachfile:
            continue
            
        logging.info(f"处理文件: {eachfile}")
        
        # 使用绝对路径加载数据
        x_train, y_train, x_test, y_test = get_data(
            pooling_type,
            is_add_noise,
            noise_ratio,
            filepath,  # 使用绝对路径
            eachfile,
            sequence_window,
            True,  # trigger_flag
            True,  # is_binary_class
            multiScale=is_multi_scale,
            waveScale=training_level,
            waveType=wave_type
        )
        
        if x_train is None:
            logging.warning(f"警告: 无法加载文件 {eachfile}")
            continue
            
        logging.info(f"成功加载文件 {eachfile}")
        logging.info(f"训练数据形状: {x_train.shape}")
        logging.info(f"测试数据形状: {x_test.shape}")