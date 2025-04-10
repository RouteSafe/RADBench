import datetime
import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

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
            if os.path.exists(os.path.join(event_path, 'metrics_old.txt')):
                os.remove(os.path.join(event_path, 'metrics_old.txt'))
            with open(os.path.join(event_path, 'pred_labels.txt')) as f:
                pred_labels.extend(eval(f.read()))
            with open(os.path.join(event_path, 'true_labels.txt')) as f:
                true_labels.extend(eval(f.read()))
        (TN, FP, FN, TP), report, precision, recall, F1_score = self._calc_metrics(true_labels, pred_labels)
        s = f"TN={TN}, FP={FP}, FN={FN}, TP={TP}\nprecision={precision}, recall={recall}, F1-Score={F1_score}\n\n{report}"
        print(s)
        with open(os.path.join(result_path, 'point-wise metrics.txt'), 'w') as f:
            f.write(s)
    

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