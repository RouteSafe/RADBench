import datetime
import os
import concurrent
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tqdm import tqdm

class Metric:
    def __init__(self, event_name, ):
        self.event_name = event_name
        # self.true_y = true_y
        # self.pred_y = pred_y

    def calculate_metrics(self, true_y, pred_y):
        save_dir = os.path.join('mydata_test_result', self.event_name)
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'true_labels.txt'), 'w') as f:
            f.write(str(true_y))
        with open(os.path.join(save_dir, 'pred_labels.txt'), 'w') as f:
            f.write(str(pred_y))
        (TN, FP, FN, TP), report, precision, recall, F1_score = self._calc_metrics(true_y, pred_y)
        # print(report)
        # print(cm)
        s = f"TN={TN}, FP={FP}, FN={FN}, TP={TP}\nprecision={precision}, recall={recall}, F1-Score={F1_score}\n\n{report}"
        print(s)
        with open(os.path.join(save_dir, 'metrics.txt'), 'w') as f:
            f.write(s)
        return (TN, FP, FN, TP), report, precision, recall, F1_score

    @ staticmethod
    def _calc_metrics(true_y, pred_y):
        report = classification_report(y_true=true_y, y_pred=pred_y,
                            target_names=['Normal', 'Abnomarl'], labels=range(2))
        # 混淆矩阵：https://zhuanlan.zhihu.com/p/350664406
        cm = confusion_matrix(y_true=true_y, y_pred=pred_y)
        TN, FP, FN, TP = cm.ravel()
        precision = TP/(TP+FP) if TP+FP else -1
        recall = TP/(TP+FN) if TP+FN else -1
        F1_score = 2*TP/(2*TP+FP+FN) if 2*TP+FP+FN else -1
        return (TN, FP, FN, TP), report, precision, recall, F1_score
    
    @staticmethod
    def _calc_auc(true_labels:list, pred_labels:list):
        # 转换为numpy数组
        true_labels = np.array(true_labels)
        pred_labels = np.array(pred_labels)
        
        # 验证输入
        if len(true_labels) != len(pred_labels):
            raise ValueError("真实标签和预测标签的长度必须相同")
        
        if len(np.unique(true_labels)) != 2:
            raise ValueError("真实标签必须是二分类的")
        
        # 使用sklearn的roc_auc_score计算AUC
        auc_score = roc_auc_score(true_labels, pred_labels)
        
        return auc_score


    # @staticmethod
    def calculate_point_wise(self, result_path, ):
        """生成所有时间的point-wise指标
        Args:
            result_path (str): 所有时间检测结果路径
        """
        pred_labels = []
        true_labels = []
        for event_name in sorted(os.listdir(result_path)):
            event_path = os.path.join(result_path, event_name)
            if not os.path.isdir(event_path):
                continue
            with open(os.path.join(event_path, 'pred_labels.txt')) as f:
                pred_labels.extend(eval(f.read()))
            with open(os.path.join(event_path, 'true_labels.txt')) as f:
                true_labels.extend(eval(f.read()))
        (TN, FP, FN, TP), report, precision, recall, F1_score = self._calc_metrics(true_labels, pred_labels)
        s = f"TN={TN}, FP={FP}, FN={FN}, TP={TP}\nprecision={precision}, recall={recall}, F1-Score={F1_score}\n\n{report}"
        print(s)
        with open(os.path.join(result_path, 'point-wise metrics.txt'), 'w') as f:
            f.write(s)

# def read_event_list(event_list_path='/data/data/anomaly-event-routedata/anomaly-event.csv', evnet_type='leak'):
#     """读取事件列表
#     Args:
#         event_list_path (str): 事件列表路径
#         type (str): 事件类型[leak, hijack, outage, ]
#     """
#     events = pd.read_csv(event_list_path)
#     events = events[events['事件类型'] == evnet_type]
#     results = []
#     for index, row in events.iterrows():
#         event_name = row['事件名称']
#         event_start_time = row['开始时间(UTC)'].strip()
#         event_end_time = row['结束时间(UTC)'].strip()
#         prefix = row['prefix']
#         hijacked_prefix = row['hijacked_prefix']
#         attack_as = row['攻击']
#         victim_as = row['受害']
#         outage = row['中断']
#         leaker_as = row['泄露']
#         attack_as = int(attack_as) if not pd.isna(attack_as) else attack_as
#         victim_as = int(victim_as) if not pd.isna(victim_as) else victim_as
#         leaker_as = int(leaker_as) if not pd.isna(leaker_as) else leaker_as

#         # if evnet_type=='leak':
#         event_start_time = datetime.datetime.strptime(event_start_time, "%Y/%m/%d %H:%M:%S").strftime("%Y-%m-%d %H:%M")
#         if event_end_time != 'unknown':
#             event_end_time = datetime.datetime.strptime(event_end_time, "%Y/%m/%d %H:%M:%S").strftime("%Y-%m-%d %H:%M")
#         data = {
#             'event_name': event_name,
#             'event_start_time': event_start_time,
#             'event_end_time': "" if event_end_time=='unknown' else event_end_time,
#             'prefix': prefix,
#             'hijacked_prefix': hijacked_prefix,
#             'attack_as': attack_as,
#             'victim_as': victim_as,
#             'outage': outage,
#             'leaker_as': leaker_as
#         }
#         # yield data
#         results.append(data)
#     return results

def read_event_list(event_list_path='/data/data/anomaly-event-routedata/anomaly-event-info.csv', evnet_type='leak'):
    """读取事件列表
    Args:
        event_list_path (str): 事件列表路径
        type (str): 事件类型[leak, hijack, outage, ]
    """
    events = pd.read_csv(event_list_path)
    events = events[events['event_type'] == evnet_type]
    results = []
    for index, row in events.iterrows():
        event_name = row['event_name']
        event_start_time = row['start_time'].strip()
        event_end_time = row['end_time'].strip()
        prefix = row['prefix']
        hijacked_prefix = row['hijacked_prefix']
        hijack_as = row['hijack_as']
        victim_as = row['vicitim_as']
        outage = row['outage_as']
        leaker_as = row['leak_as']
        hijack_as = int(hijack_as) if not pd.isna(hijack_as) else hijack_as
        victim_as = int(victim_as) if not pd.isna(victim_as) else victim_as
        leaker_as = int(leaker_as) if not pd.isna(leaker_as) else leaker_as

        # if evnet_type=='leak':
        event_start_time = datetime.datetime.strptime(event_start_time, "%Y/%m/%d %H:%M").strftime("%Y-%m-%d %H:%M")
        if event_end_time != 'unknown':
            event_end_time = datetime.datetime.strptime(event_end_time, "%Y/%m/%d %H:%M").strftime("%Y-%m-%d %H:%M")
        data = {
            'event_name': event_name,
            'event_start_time': event_start_time,
            'event_end_time': "" if event_end_time=='unknown' else event_end_time,
            'prefix': prefix,
            'hijacked_prefix': hijacked_prefix,
            'hijack_as': hijack_as,
            'victim_as': victim_as,
            'outage': outage,
            'leaker_as': leaker_as
        }
        # yield data
        results.append(data)
    return results

def decodeMRT(filepaths:list):
    if not isinstance(filepaths, list):
        filepaths = [filepaths]
    all_data = []
    # print(f"decoding MRT files......{filepaths}")
    for filepath in filepaths:
        decoded_data = os.popen(f"bgpdump -q -m {filepath}").read()
        decoded_data = decoded_data.strip('\n').split("\n")
        all_data.extend(decoded_data)
    return all_data


def decodeMRT_warp(x):
    filepath, savepath = x
    decoded_data = decodeMRT(filepath)
    with open(savepath, 'w') as f:
        f.write('\n'.join(decoded_data))

def decode_all_dataset():
    root_path = '/data/data/anomaly-event-routedata'
    dirs = os.listdir(root_path)
    dirs = [i for i in dirs if os.path.isdir(os.path.join(root_path, i))]
    all_tasks = []
    for dirname in dirs:
        dir_path = os.path.join(root_path, dirname)
        decoded_dirname = 'decoded'
        if os.path.exists(os.path.join(dir_path, decoded_dirname)) and dirname.startswith('leak'):
            decoded_dirname = 'decoded_new'
        os.makedirs(os.path.join(dir_path, decoded_dirname), exist_ok=True)
        for filename in os.listdir(dir_path):
            if filename.endswith('.gz'):
                filepath = os.path.join(dir_path, filename)
                savepath = os.path.join(dir_path, decoded_dirname, filename.replace('.gz', '.txt'))
                if not os.path.exists(savepath) or os.path.getsize(savepath) == 0:
                    all_tasks.append((filepath, savepath))
    all_tasks = sorted(all_tasks, key=lambda x: x[0])
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=32) as executor:
        results = list(tqdm(executor.map(decodeMRT_warp, all_tasks), total=len(all_tasks)))
    return results

def f():
    root_path = '/data/data/anomaly-event-routedata'
    dirs = os.listdir(root_path)
    dirs = [i for i in dirs if os.path.isdir(os.path.join(root_path, i)) and i.startswith('leak')]
    dirs = sorted(dirs)
    for dirname in dirs:
        if not dirname.startswith('leak'):
            continue
        dir_path = os.path.join(root_path, dirname)
        os.rename(src=os.path.join(dir_path, 'decoded'), dst=os.path.join(dir_path, 'decoded_old'))
        os.rename(src=os.path.join(dir_path, 'decoded_new'), dst=os.path.join(dir_path, 'decoded'))

        

if __name__ == '__main__':
    # f()
    decode_all_dataset()
    # read_event_list()