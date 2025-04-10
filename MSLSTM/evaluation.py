import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve,auc
import sklearn
import loaddata_mydata
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def ComputeAUC(y_test,y_predict):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_predict[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_predict.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    #Plot of a ROC curve for a specific class
    #return (fpr[0]+fpr[1])/2,(tpr[0]+tpr[1])/2,(roc_auc[0]+roc_auc[1])/2
    return fpr[0],tpr[0],roc_auc[0]




def evaluation(label,predict,trigger_flag,evalua_flag,positive_label=1,negative_label=0):
    """评估模型性能"""
    ac_positive = 0
    ac_negative = 0
    correct = 0
    
    # 转换为numpy数组
    label = np.array(label)
    if len(label.shape) > 1 and label.shape[1] > 1:
        label = np.argmax(label, axis=1)
    predict = np.array(predict)
    if len(predict.shape) > 1 and predict.shape[1] > 1:
        predict = np.argmax(predict, axis=1)
    
    # 计算正例和负例数量
    predict_positive = np.sum(predict == positive_label)
    total_positive = np.sum(label == positive_label)
    predict_negative = np.sum(predict == negative_label)
    total_negative = np.sum(label == negative_label)

    # 使用numpy的比较操作
    correct_predictions = (predict == label)
    correct = np.sum(correct_predictions)
    logging.info(f"TP+TN: {correct}")
    
    # 分别计算正确预测的正例和负例
    ac_positive = np.sum((label == positive_label) & (predict == label))
    logging.info(f"TP: {ac_positive}")
    ac_negative = np.sum((label == negative_label) & (predict == label))
    logging.info(f"TN: {ac_negative}")
    try:
        # 添加除零检查
        PRECISION = float(ac_positive) / predict_positive if predict_positive > 0 else 0
        # 修改AUC计算方式
        try:
            AUC = roc_auc_score(label, predict, multi_class='ovr')
        except ValueError:
            # 如果是二分类问题，不需要指定multi_class
            AUC = roc_auc_score(label, predict)
            
        G_MEAN = np.sqrt(float(ac_positive * ac_negative) / (total_negative * total_positive)) if (total_negative * total_positive) > 0 else 0
        RECALL = float(ac_positive) / total_positive if total_positive > 0 else 0
        
        # 添加除零检查
        if PRECISION + RECALL > 0:
            F1_SCORE = round((2 * PRECISION * RECALL) / (PRECISION + RECALL), 5)
        else:
            F1_SCORE = 0

        return {
            "AUC": AUC,
            "G_MEAN": G_MEAN,
            "RECALL": RECALL,
            "PRECISION": PRECISION,
            "F1_SCORE": F1_SCORE
        }

    except Exception as e:
        print(f"计算评价指标时出错: {str(e)}")

def evaluation2(label,predict,trigger_flag,evalua_flag,positive_label=1,negative_label=0):

    if trigger_flag or evalua_flag == False:
        Output_Class = [[] for i in range(len(label[0]))]
        if len(predict) == len(label):
            for tab_sample in range(len(label)):
                max_index = predict[tab_sample].argmax(axis=0)
                for tab_class in range(len(Output_Class)):
                    if tab_class == max_index:
                        Output_Class[tab_class].append(1)
                    else:
                        Output_Class[tab_class].append(0)
        else:
            print("Error!")
    else:
        Output_Class = []
        for tab_sample in range(len(predict)):
            Output_Class.append(int(predict[tab_sample]))

    ac_positive = 0
    ac_negative = 0
    correct = 0
    error_rate_flag = False
    if trigger_flag or evalua_flag == False:
        for tab_class in range(len(Output_Class)):
            error_rate_flag = True
            if label[0][tab_class] == negative_label:
                predict_negative = Output_Class[tab_class].count(1)
                total_negative = list(np.transpose(label)[tab_class]).count(1)

                for tab_sample in range(len(label)):
                    if Output_Class[tab_class][tab_sample]==1 and Output_Class[tab_class][tab_sample] == int(label[tab_sample][tab_class]) :
                        ac_negative += 1
            elif label[0][tab_class] == positive_label:
                predict_positive = Output_Class[tab_class].count(1)
                total_positive = list(np.transpose(label)[tab_class]).count(1)
                for tab_sample in range(len(label)):
                    if Output_Class[tab_class][tab_sample]==1 and Output_Class[tab_class][tab_sample] == int(label[tab_sample][tab_class]):
                        ac_positive += 1
    else:
        #output class error rate
        predict_positive = Output_Class.count(positive_label)
        total_positive = list(label).count(positive_label)

        predict_negative = Output_Class.count(negative_label)
        total_negative = list(label).count(negative_label)
        for tab_sample in range(len(predict)):
            if predict[tab_sample] == label[tab_sample]:
                correct += 1
                if int(label[tab_sample]) == positive_label:
                    ac_positive += 1
                elif int(label[tab_sample]) == negative_label:
                    ac_negative += 1

    #if error_rate_flag == True:
        #print("Error Rate is :"+str((len(Output_Class) - correct)/float(len(Output_Class))))
        #return {"Error_Rate":(len(Output_Class) - correct)/float(len(Output_Class))}

    #PlottingAUC(ReverseEncoder(label),ReverseEncoder(np.transpose(np.array(Output_Class))))
    if evalua_flag == True:
        try:
            ACC_A = float(ac_positive) / predict_positive
        except:
            ACC_A = float(ac_positive) * 100 / (predict_positive + 1)

        AUC = roc_auc_score(label, np.transpose(np.array(Output_Class)))
        G_MEAN = np.sqrt(float(ac_positive * ac_negative) / (total_negative * total_positive))
        FPR,TPR,AUC = ComputeAUC(label,predict)
        
        # 计算 RECALL, PRECISION 和 F1_SCORE
        RECALL = float(ac_positive) / total_positive
        try:
            PRECISION = float(ac_positive) / predict_positive
        except:
            PRECISION = float(ac_positive) * 100 / (predict_positive + 1)
            
        try:
            F1_SCORE = round((2 * PRECISION * RECALL) / (PRECISION + RECALL), 5)
        except:
            F1_SCORE = 0.01 * round((2 * PRECISION * RECALL) / (PRECISION + RECALL + 1), 5)
            
        print({"FPR":FPR,"TPR":TPR,"AUC":AUC,"G_MEAN":G_MEAN,"PRECISION":PRECISION,"RECALL":RECALL,"F1_SCORE":F1_SCORE})
        return {"AUC":AUC,"G_MEAN":G_MEAN,"RECALL":RECALL,"PRECISION":PRECISION,"F1_SCORE":F1_SCORE}


    else:
        AUC = roc_auc_score(label, np.transpose(np.array(Output_Class)))
        G_MEAN = np.sqrt(float(ac_positive * ac_negative) / (total_negative * total_positive))
        FPR,TPR,AUC = ComputeAUC(label,predict)
        
        # 计算 RECALL, PRECISION 和 F1_SCORE
        RECALL = float(ac_positive) / total_positive
        try:
            PRECISION = float(ac_positive) / predict_positive
        except:
            PRECISION = float(ac_positive) * 100 / (predict_positive + 1)
            
        try:
            F1_SCORE = round((2 * PRECISION * RECALL) / (PRECISION + RECALL), 5)
        except:
            F1_SCORE = 0.01 * round((2 * PRECISION * RECALL) / (PRECISION + RECALL + 1), 5)
            
        print({"FPR":FPR,"TPR":TPR,"AUC":AUC,"G_MEAN":G_MEAN,"PRECISION":PRECISION,"RECALL":RECALL,"F1_SCORE":F1_SCORE})
        return {"AUC":AUC,"G_MEAN":G_MEAN,"RECALL":RECALL,"PRECISION":PRECISION,"F1_SCORE":F1_SCORE}
        #return fpr,tpr,auc#evaluate auc




