# -*- coding:utf-8 -*-
from __future__ import division
import sys
import printlog
import datetime
import os
import time
import sklearn
from sklearn.metrics import confusion_matrix
from baselines import sclearn
import evaluation
from pathlib import Path
from collections import defaultdict
import tensorflow as tf
import mslstm_mydata
import config
import loaddata_mydata
import numpy as np
import visualize
from sklearn.metrics import accuracy_score
from baselines import nnkeras_mydata,sclearn_mydata
import matplotlib.pyplot as plt
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('train_mydata_lstm.log')
    ]
)

flags = tf.app.flags
FLAGS = flags.FLAGS


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]

    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)

    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
def pprint(msg,method=''):
    #if not 'Warning' in msg:
    if 'Warning' in msg:
        sys.stdout = printlog.PyLogger('',method+'_'+str(FLAGS.num_neurons1))
        print(msg)
        try:
            sys.stderr.write(msg+'\n')
        except:
            pass
        #sys.stdout.flush()
    else:
        print(msg)
#def sess_run(commander,data,label):
    #global sess, data_x, data_y
    #return sess.run(commander, {data_x: data, data_y: label})

def train_lstm(method, filename_test, trigger_flag, evalua_flag, is_binary_class):
    # 使用绝对路径
    data_dir = FLAGS.data_dir
    
    try:
        global positive_sign, negative_sign
        positive_sign = 1
        negative_sign = 0
        x_train, y_train, x_val, y_val = loaddata_mydata.get_data(
            FLAGS.pooling_type, 
            FLAGS.is_add_noise, 
            FLAGS.noise_ratio, 
            data_dir,
            filename_test, 
            FLAGS.sequence_window, 
            trigger_flag,
            is_binary_class,
            multiScale=FLAGS.is_multi_scale, 
            waveScale=FLAGS.scale_levels,
            waveType=FLAGS.wave_type
        )
        
        # 如果需要验证集，可以从训练数据中分割
        train_size = int(0.8 * len(x_train))
        x_val = x_train[train_size:]
        y_val = y_train[train_size:]
        x_train = x_train[:train_size]
        y_train = y_train[:train_size]
        
        logging.info(f"训练数据形状: x_train={x_train.shape}, y_train={y_train.shape}")
        logging.info(f"验证数据形状: x_val={x_val.shape}, y_val={y_val.shape}")
        
    except Exception as e:
        logging.error(f"加载数据出错: {str(e)}")
        return None

    # 检查数据是否成功加载
    if x_train is None or y_train is None:
        logging.error("训练数据加载失败")
        return None
    
    global tempstdout
    FLAGS.option = method
    
    if FLAGS.is_multi_scale:
        FLAGS.scale_levels = x_train.shape[1]
        FLAGS.input_dim = x_train.shape[-1]
        FLAGS.number_class = y_train.shape[1]
    else:
        FLAGS.input_dim = x_train.shape[-1]
        FLAGS.number_class = y_train.shape[1]

    with tf.Graph().as_default():
        config = tf.ConfigProto(device_count={'GPU': 0})
        tf.set_random_seed(1337)
        
        data_x, data_y = mslstm_mydata.inputs(FLAGS.option)
        is_training = tf.placeholder(tf.bool)
        prediction, label, output_last = mslstm_mydata.inference(data_x, data_y, FLAGS.option, is_training)
        loss, recall, precision = mslstm_mydata.loss_(prediction, label)
        train_op, optimizer = mslstm_mydata.train(loss, recall, precision)
        minimize = optimizer.minimize(loss)
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # 初始化变量
        init_op = tf.global_variables_initializer()
        sess = tf.Session(config=config)
        sess.run(init_op)

        # 创建保存目录
        model_save_dir = os.path.join('/data/data/xiaolan_data/MSLSTM', "tf_tmp")
        os.makedirs(model_save_dir, exist_ok=True)
        
        # 创建saver
        saver = tf.train.Saver()

        # 训练循环
        epoch_training_loss_list = []
        epoch_training_acc_list = []
        epoch_val_loss_list = []
        epoch_val_acc_list = []
        early_stopping = 10
        total_iteration = 0

        for i in range(FLAGS.max_epochs):
            if early_stopping <= 0:
                break
                
            j_iteration = 0
            for j_batch in iterate_minibatches(x_train, y_train, FLAGS.batch_size, shuffle=False):
                j_iteration += 1
                total_iteration += 1
                inp, out = j_batch
                
                # 训练步骤
                sess.run(minimize, {data_x: inp, data_y: out, is_training: True})
                training_acc, training_loss = sess.run((accuracy, loss), 
                                                     {data_x: inp, data_y: out, is_training: True})
                
                # 验证步骤
                val_acc, val_loss = sess.run((accuracy, loss), 
                                           {data_x: x_val, data_y: y_val, is_training: False})

            # 记录每个epoch的结果
            logging.info(
                f"{FLAGS.option}_Epoch{i+1}>_Titer-{total_iteration}_iter-{j_iteration}"
                f"{FLAGS.wave_type}-{FLAGS.scale_levels}-{FLAGS.learning_rate}-"
                f"{FLAGS.num_neurons1}-{FLAGS.num_neurons2}>>>="
                f"train_accuracy: {training_acc:.4f}, train_loss: {training_loss:.4f}"
                f",\tval_accuracy: {val_acc:.4f}, val_loss: {val_loss:.4f}"
            )

            # 保存指标
            epoch_training_loss_list.append(training_loss)
            epoch_training_acc_list.append(training_acc)
            epoch_val_loss_list.append(val_loss)
            epoch_val_acc_list.append(val_acc)

            # 早停检查
            try:
                max_val_acc = epoch_val_acc_list[-2]
            except:
                max_val_acc = 0

            if epoch_val_acc_list[-1] < max_val_acc:
                early_stopping -= 1
            elif epoch_val_acc_list[-1] >= max_val_acc:
                early_stopping = 10
                
            if val_loss > 10 or np.isnan(val_loss):
                break

        # 创建一个模型类来封装预测功能
        class LSTMModel:
            def __init__(self, session, data_x, data_y, prediction, is_training):
                self.session = session
                self.data_x = data_x
                self.data_y = data_y
                self.prediction = prediction
                self.is_training = is_training
                self.expected_shape = data_x.get_shape().as_list()
                logging.info(f"模型期望的输入形状: {self.expected_shape}")
            
            def predict(self, x_test, y_test):
                try:
                    if self.session is None:
                        logging.error("Session is None")
                        return None
                        
                    logging.info(f"开始预测，输入数据形状: {x_test.shape}")
                    pred = self.session.run(
                        self.prediction, 
                        {
                            self.data_x: x_test, 
                            self.data_y: y_test,
                            self.is_training: False
                        }
                    )
                    logging.info(f"预测完成，输出形状: {pred.shape}")
                    return pred
                except Exception as e:
                    logging.error(f"预测时出错: {str(e)}")
                    return None

        # 保存模型但不关闭session
        saver.save(sess, os.path.join(model_save_dir, "model.ckpt"))
        
        # 返回模型对象
        model = LSTMModel(sess, data_x, data_y, prediction, is_training)
        logging.info(f"LSTM模型 {method} 创建成功")
        return model

def train_classic(method, filename_test, trigger_flag, evalua_flag, is_binary_class):
    logging.info(f"开始训练经典模型 {method}")
    try:
        model = sclearn_mydata.Basemodel(method, filename_test, trigger_flag)
        if model is None:
            logging.error(f"训练经典模型 {method} 失败")
        else:
            logging.info(f"经典模型 {method} 训练成功")
        return model
    except Exception as e:
        logging.error(f"训练经典模型时出错: {str(e)}")
        return None

def train(method, filename_test, trigger_flag, evalua_flag, is_binary_class, wave_type='db1'):
    logging.info(f'开始训练模型 {method}')
    if 'L' in method or 'RNN' in method:
        sys.stdout = tempstdout
        if method == '1L' or method == '2L' or method == '3L' \
                or method == '4L' or method == '5L' or method == 'RNN':
            FLAGS.is_multi_scale = False
        elif 'AL' == method:
            FLAGS.is_multi_scale = False
            logging.info("AL模型: 设置 is_multi_scale = False")
        else:
            FLAGS.is_multi_scale = True
            FLAGS.wave_type = wave_type
            
        logging.info(f"调用 train_lstm 训练模型 {method}")
        model = train_lstm(method, filename_test, trigger_flag, evalua_flag, is_binary_class)
        if model is None:
            logging.error(f"LSTM模型 {method} 训练失败")
        else:
            logging.info(f"LSTM模型 {method} 训练成功")
        return model
    else:
        sys.stdout = tempstdout
        model = train_classic(method, filename_test, trigger_flag, evalua_flag, is_binary_class)
        if model is None:
            logging.error(f"经典模型 {method} 训练失败")
        return model

def get_test_files(data_dir, train_files):
    """获取测试文件列表"""
    try:
        all_files = [f for f in os.listdir(data_dir)]
        test_files = [f for f in all_files if f not in train_files and (f.startswith('hijack') or f.startswith('leak') or f.startswith('outage'))]
        return test_files
    except Exception as e:
        logging.error(f"获取测试文件列表时出错: {str(e)}")
        return []

def save_evaluation_results(results, output_file, combination_id, model_configs):
    """保存评估结果到CSV文件，添加组合ID列和事件类型列"""
    try:
        # 首先检查results是否为空
        if not results:
            logging.error("评估结果为空")
            return None
            
        rows = []
        for model_name, model_results in results.items():
            if not model_results:  # 检查model_results是否为空
                logging.warning(f"模型 {model_name} 没有评估结果")
                continue
                
            for file_name, metrics in model_results.items():
                if not metrics:  # 检查metrics是否为空
                    logging.warning(f"文件 {file_name} 的评估指标为空")
                    continue
                    
                try:
                    # 从文件名提取事件类型（第一个'-'之前的部分）
                    event_type = file_name.split('-')[0]
                    
                    row = {
                        'Combination': f'组合{combination_id}',
                        'Model': model_name,
                        'Type': event_type,  # 新增type列
                        'File': file_name,
                        'AUC': metrics['AUC'],
                        'G-Mean': metrics['G_MEAN'],
                        'Recall': metrics['RECALL'],
                        'Precision': metrics['PRECISION'],
                        'F1-Score': metrics['F1_SCORE'],
                        'Learning_Rate': model_configs[model_name]['learning_rate'],
                        'Neurons1': model_configs[model_name]['neurons1'],
                        'Neurons2': model_configs[model_name]['neurons2']
                    }
                    rows.append(row)
                except KeyError as e:
                    logging.error(f"评估指标缺失: {str(e)}")
                    continue
                except Exception as e:
                    logging.error(f"处理文件名 {file_name} 时出错: {str(e)}")
                    continue
        
        # 检查是否有有效的评估结果
        if not rows:
            logging.error("没有有效的评估结果可以保存")
            return None
            
        # 创建DataFrame
        df = pd.DataFrame(rows)
        logging.info(f"成功创建评估结果DataFrame，包含 {len(rows)} 条记录")
        return df
        
    except Exception as e:
        logging.error(f"处理评估结果时出错: {str(e)}")
        return None

def predict_with_shape_check(model, x_test, y_test, sequence_window, input_dim, need_window=False):
    """检查数据形状并进行预测"""
    try:
        # 如果需要窗口化处理（LSTM相关模型）
        if need_window:
            # 检查输入维度
            if x_test.shape[-1] != input_dim:
                logging.error(f"特征维度不匹配: 期望 {input_dim}, 实际 {x_test.shape[-1]}")
                return None
                
            # 应用滑动窗口
            if len(x_test.shape) == 2:
                x_test, y_test = loaddata_mydata.slidingFunc(sequence_window, x_test, y_test)
                logging.info(f"应用滑动窗口后形状: x_test={x_test.shape}, y_test={y_test.shape}")
            
            # 进行预测
            y_pred = model.predict(x_test, y_test)
            return y_pred
        else:
            return model.predict(x_test)
        
    except Exception as e:
        logging.error(f"预测过程出错: {str(e)}")
        return None

def main(unused_argv):
    # 手动指定要使用的文件组合
    combinations = [
        # 第一组8个文件
        [
            "hijack-20080224-Pakistan_Telecom_hijacked_YouTube.txt",
            "hijack-20080810-Pilosoft_hijacked_Sparkplug_Las_Vegas.txt",
            "hijack-20110114-Indonesian_Indosat_hijack2011.txt",
            "hijack-20170426-PJSC_Rostelecom.txt",
            "leak-20041224-TTNet_in_Turkey_leak.txt",
            "leak-20160422-AWS_Route_Leak.txt",
            "leak-20210211-Cablevision_Mexico_leak",
            "outage-20110311-Japan_Earthquake.txt"
        ],
        # 第二组8个文件
        [
            "hijack-20140402-Indonesian_Indosat_hijack2014.txt",
            "hijack-20150107-The_IP_squatters_hijacked_Italian_National_Institute_of_Nuclear_Physics.txt",
            "hijack-20151106-India_BHARTI_Airtel_hijack.txt",
            "hijack-20170426-PJSC_Rostelecom.txt",
            "leak-20171106-Level_3_leak.txt",
            "leak-20150612-Malaysian_Telecom_leak.txt",
            "leak-20170825-Google_leak.txt",
            "outage-20211004-Facebook_outage.txt"
        ],
        # 第三组8个文件
        [
            "hijack-20151204-BackConnect_hijacked_VolumeDrive.txt",
            "hijack-20220215-Nigeria_JSC_Ukraine.txt",
            "hijack-20220228-Russia_Ukraine.txt",
            "leak-20190606-SafeHost_leak.txt",
            "leak-20241030-Worldstream_leak.txt",
            "leak-20200721-Commercial_Conecte_leak.txt",
            "outage-20220313-RU_BGP_outage.txt",
            "outage-20211004-Facebook_outage.txt"
        ],
        # 第四组8个文件
        [
            "hijack-20080224-Pakistan_Telecom_hijacked_YouTube.txt",
            "hijack-20160221-BackConnect_GHOSTnet.txt",
            "hijack-20180629-Bitcanal_Jingdong.txt",
            "hijack-20141114-H3S_median_services_hijacked_Comcast.txt",
            "leak-20150612-Malaysian_Telecom_leak.txt",
            "leak-20120223-Australia_Telstra_leak.txt",
            "leak-20190606-SafeHost_leak.txt",
            "outage-20211109-Comcast_outages_1.txt"
        ],
        # 第五组8个文件
        [
            "hijack-20140402-Indonesian_Indosat_hijack2014.txt",
            "hijack-20220215-Nigeria_JSC_Ukraine.txt",
            "hijack-20220228-Russia_Ukraine.txt",
            "leak-20150612-Malaysian_Telecom_leak.txt",
            "leak-20181112-Nigeria_MainOne_Cable_leak.txt",
            "outage-20110311-Japan_Earthquake.txt",
            "outage-20210113-Uganda_election_shutdown.txt",
            "outage-20211109-Comcast_outages_1.txt"
        ]]

    
    # 创建一个列表存储所有组合的结果
    all_combinations_results = []
    
    for i, train_files in enumerate(combinations):
        logging.info(f"\n开始处理第 {i+1} 个组合:")
        logging.info(f"训练文件: {train_files}")
        
        # 验证训练文件是否存在
        feature_dir = Path(FLAGS.data_dir)
        all_files_exist = True
        for train_file in train_files:
            if not (feature_dir / train_file).exists():
                logging.error(f"训练文件不存在: {train_file}")
                all_files_exist = False
                break
        
        if not all_files_exist:
            continue
            
        # 合并训练数据
        def load_and_combine_data(files, combination_id):
            combined_data = None
            for file in files:
                try:
                    data = loaddata_mydata.LoadData(FLAGS.data_dir, file)
                    if data is not None:
                        # 确保数据是数值类型
                        data = np.array(data, dtype=np.float64)
                        if combined_data is None:
                            combined_data = data
                        else:
                            combined_data = np.concatenate((combined_data, data), axis=0)
                        logging.info(f"成功加载文件: {file}, 形状: {data.shape}")
                except Exception as e:
                    logging.error(f"加载文件 {file} 时出错: {str(e)}")
            
            # 保存合并后的数据，使用组合编号命名
            if combined_data is not None:
                try:
                    output_file = os.path.join(FLAGS.data_dir, f"combined_8_data_model_{combination_id}.txt")
                    # 使用numpy的savetxt保存数据，指定格式为1位小数
                    np.savetxt(
                        output_file,
                        combined_data,
                        delimiter=",",
                        fmt='%.1f'  # 保留1位小数
                    )
                    logging.info(f"合并数据已保存到: {output_file}")
                    logging.info(f"合并数据形状: {combined_data.shape}")
                except Exception as e:
                    logging.error(f"保存合并数据时出错: {str(e)}")
            
            return os.path.basename(output_file)

        combined_train_data = load_and_combine_data(train_files, i+1)

        if combined_train_data is None:
            logging.error("合并训练数据失败")
            continue
        
        logging.info(f"合并训练数据成功: {combined_train_data}")

        # 设置固定参数
        FLAGS.sequence_window = 30  # 滑动窗口大小
        FLAGS.input_dim = 24       # 特征维度
        FLAGS.number_class = 2     # 类别数
        
        # 设置模型参数
        trigger_flag = 1
        evalua_flag = True
        is_binary_class = True
        wave_type = 'haar'

        case_label = {'SVM':'SVM','NB':'NB','DT':'DT','Ada.Boost':'Ada.Boost','RF':'RF','1NN':'1NN','1NN-DTW':'DTW',
                      'SVMF':'SVMF','SVMW':'SVMW','MLP':'MLP','RNN':'RNN','1L':'LSTM','2L':'2-LSTM','3L':'3-LSTM',\
                      'AL':'ALSTM','HL':'MSLSTM','HAL':'MSLSTM'}

        if trigger_flag == 1:
            case = ["RNN", "HAL"]
        
        # 设置神经网络参数
        hidden_unit1_list = [8, 16]
        hidden_unit2_list = [8]
        learning_rate_list = [0.01,0.001]

        # 训练每个模型
        case_list = []
        # 获取测试文件列表
        test_files = get_test_files(FLAGS.data_dir, train_files)
        if not test_files:
            logging.error("未找到测试文件")
            continue
        
        # 存储所有评估结果
        all_results = defaultdict(dict)
        model_configs = {}  # 存储不同配置的模型信息
        
        for each_case in case:
            case_list.append(case_label[each_case])
            
            for learning_rate in learning_rate_list:
                FLAGS.learning_rate = learning_rate
                # 根据模型类型选择神经元配置
                if 'H' in each_case:  # 多尺度模型 (HL, HAL)
                    for neurons1 in hidden_unit1_list:
                        for neurons2 in hidden_unit2_list:
                            FLAGS.num_neurons1 = neurons1
                            FLAGS.num_neurons2 = neurons2
                            # 创建配置ID
                            config_id = f"{each_case}_lr{learning_rate}_n1_{neurons1}_n2_{neurons2}"
                            
                            # 保存模型配置信息
                            model_configs[config_id] = {
                                'learning_rate': learning_rate,
                                'neurons1': neurons1,
                                'neurons2': neurons2
                            }
                            
                            logging.info(f'训练模型配置: {config_id}')
                            model = train(
                                each_case,
                                combined_train_data,
                                trigger_flag,
                                evalua_flag,
                                is_binary_class,
                                wave_type
                            )
                            # 在每个测试文件上评估模型
                            for test_file in test_files:
                                logging.info(f"测试文件: {test_file}")
                                x_test, y_test = loaddata_mydata.get_data_withoutval(
                                    FLAGS.pooling_type, 
                                    FLAGS.is_add_noise, 
                                    FLAGS.noise_ratio, 
                                    FLAGS.data_dir,
                                    test_file, 
                                    FLAGS.sequence_window, 
                                    trigger_flag,
                                    is_binary_class,
                                    multiScale=FLAGS.is_multi_scale, 
                                    waveScale=FLAGS.scale_levels,
                                    waveType=FLAGS.wave_type
                                )
                                try:
                                    # 进行预测，LSTM相关模型需要窗口化
                                    y_pred = predict_with_shape_check(
                                        model, 
                                        x_test, 
                                        y_test,
                                        FLAGS.sequence_window,
                                        FLAGS.input_dim,
                                        need_window=True  # LSTM相关模型需要窗口化
                                    )
                                    if y_pred is None:
                                        logging.error("预测返回None")
                                        continue
                                        
                                    logging.info(f"原始预测结果形状: {y_pred.shape}")
                                    
                                    # 确保形状匹配
                                    assert len(y_pred) == len(y_test), f"预测结果长度 {len(y_pred)} 与标签长度 {len(y_test)} 不匹配"
                                    
                                except Exception as e:
                                    logging.error(f"预测过程出错: {str(e)}")
                                    continue
                                # 计算评估指标
                                try:
                                    metrics = evaluation.evaluation(y_test, y_pred, trigger_flag=False, evalua_flag=True)
                                    # 使用config_id存储结果
                                    all_results[config_id][test_file] = metrics
                                    
                                    logging.info(f"\n配置 {config_id} 在文件 {test_file} 的评估结果:")
                                    logging.info(f"AUC: {metrics['AUC']:.4f}")
                                    logging.info(f"G-Mean: {metrics['G_MEAN']:.4f}")
                                    logging.info(f"Recall: {metrics['RECALL']:.4f}")
                                    logging.info(f"Precision: {metrics['PRECISION']:.4f}")
                                    logging.info(f"F1-Score: {metrics['F1_SCORE']:.4f}")
                                except Exception as e:
                                    logging.error(f"计算评估指标时出错: {str(e)}")
                                    continue
                else: # 单层模型
                    for neurons1 in hidden_unit1_list: 
                        FLAGS.num_neurons1 = neurons1
                        # 创建配置ID
                        config_id = f"{each_case}_lr{learning_rate}_n1_{FLAGS.num_neurons1}"
                        
                        # 保存模型配置信息
                        model_configs[config_id] = {
                            'learning_rate': learning_rate,
                            'neurons1': FLAGS.num_neurons1,
                            'neurons2': FLAGS.num_neurons2 if 'H' in each_case else None
                        }
                    
                        logging.info(f'训练模型配置: {config_id}')
                        model = train(
                            each_case,
                            combined_train_data,
                            trigger_flag,
                            evalua_flag,
                            is_binary_class,
                            wave_type
                        )
                        logging.info(f"模型配置 {config_id} 训练成功，开始测试")
                    
                        # 在每个测试文件上评估模型
                        for test_file in test_files:
                            logging.info(f"测试文件: {test_file}")
                            x_test, y_test = loaddata_mydata.get_data_withoutval(
                                FLAGS.pooling_type, 
                                FLAGS.is_add_noise, 
                                FLAGS.noise_ratio, 
                                FLAGS.data_dir,
                                test_file, 
                                FLAGS.sequence_window, 
                                trigger_flag,
                                is_binary_class,
                                multiScale=FLAGS.is_multi_scale, 
                                waveScale=FLAGS.scale_levels,
                                waveType=FLAGS.wave_type
                            )
                            
                            # 根据模型类型选择不同的预测方式
                            if 'L' in each_case or 'RNN' in each_case:
                                # 进行预测，LSTM相关模型需要窗口化
                                y_pred = predict_with_shape_check(
                                    model, 
                                    x_test, 
                                    y_test,
                                    FLAGS.sequence_window,
                                    FLAGS.input_dim,
                                    need_window=True  # LSTM相关模型需要窗口化
                                )
                                logging.info(f"原始预测结果形状: {y_pred.shape}")
                                    
                                    # 确保形状匹配
                                assert len(y_pred) == len(y_test), f"预测结果长度 {len(y_pred)} 与标签长度 {len(y_test)} 不匹配"
                                    
                            # 计算评估指标
                            try:
                                metrics = evaluation.evaluation(y_test, y_pred, trigger_flag=False, evalua_flag=True)
                                # 使用config_id存储结果
                                all_results[config_id][test_file] = metrics
                                
                                logging.info(f"\n配置 {config_id} 在文件 {test_file} 的评估结果:")
                                logging.info(f"AUC: {metrics['AUC']:.4f}")
                                logging.info(f"G-Mean: {metrics['G_MEAN']:.4f}")
                                logging.info(f"Recall: {metrics['RECALL']:.4f}")
                                logging.info(f"Precision: {metrics['PRECISION']:.4f}")
                                logging.info(f"F1-Score: {metrics['F1_SCORE']:.4f}")
                            except Exception as e:
                                logging.error(f"计算评估指标时出错: {str(e)}")
                                continue
            
        # 每个模型类型的所有配置评估完成后，保存结果
        if all_results:
            df = save_evaluation_results(all_results, None, i+1, model_configs)
            if df is not None and not df.empty:
                all_combinations_results.append(df)
                logging.info(f"成功添加组合 {i+1} 的评估结果")
            else:
                logging.warning(f"组合 {i+1} 没有有效的评估结果")
        else:
            logging.warning(f"组合 {i+1} 没有产生任何评估结果")

    # 合并所有组合的结果
    if all_combinations_results:
        final_df = pd.concat(all_combinations_results, ignore_index=True)
        
        parent_dir = Path(FLAGS.data_dir).parent
        result_dir = parent_dir / "test_result_mydata"
        result_dir.mkdir(exist_ok=True)  
        
        # 保存详细结果
        output_file = os.path.join(result_dir, "all_combinations_results_lstm.csv")
        final_df.to_csv(output_file, index=False, float_format='%.4f')
        logging.info(f"\n所有组合的评估结果已保存到: {output_file}")
        
        # 按更多维度进行汇总统计
        summary = final_df.groupby(['Combination', 'Model', 'Learning_Rate', 'Neurons1', 'Neurons2'])[
            ['AUC', 'G-Mean', 'Recall', 'Precision', 'F1-Score']
        ].mean()
        
        summary_file = os.path.join(result_dir, "all_combinations_results_summary_lstm.csv")
        summary.to_csv(summary_file, float_format='%.4f')
        logging.info(f"评估结果汇总已保存到: {summary_file}")

    end = time.time()
    pprint("The time elapsed: " + str(end - start) + ' seconds.\n')

if __name__ == "__main__":
    logging.info("程序启动")
    global tempstdout
    tempstdout = sys.stdout
    logging.info(f"当前时间: {datetime.datetime.now()}")
    start = time.time()
    try:
        tf.app.run()
    except Exception as e:
        logging.error(f"程序执行出错: {str(e)}", exc_info=True)
    finally:
        end = time.time()
        logging.info(f"程序结束，总运行时间: {end - start} 秒")