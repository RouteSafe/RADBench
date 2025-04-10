import copy
import gc
import json
import os
import datetime
import numpy as np
import requests
import bz2
import logging
import networkx as nx
from commons import Metric, read_event_list
from tqdm import tqdm
logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S', format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

rel_file_root = './rel_data'
# events_root = './events'
events_root = '/data/data/wangyuye_data/BgpAnomalyData/leak'

def get_rel_filepath(source, event_start_date:datetime.datetime):
    rel_dataset_path = os.path.join(rel_file_root, source)
    os.makedirs(rel_dataset_path, exist_ok=True)
    if source == "CAIDA":
        root_url = "https://publicdata.caida.org/datasets/as-relationships"
        date1 = event_start_date.replace(day=1)
        if date1<datetime.datetime(year=2015, month=12, day=1):
            filename = f"{date1.strftime("%Y%m%d")}.as-rel.txt.bz2"
            url = f'{root_url}/serial-1/{filename}'
        else:
            filename = f"{date1.strftime("%Y%m%d")}.as-rel2.txt.bz2"
            url = f'{root_url}/serial-2/{filename}'
        txt_filename = filename.removesuffix('.bz2')
        if not os.path.exists(os.path.join(rel_dataset_path, txt_filename)):
            log.info(f'Downloading {filename} from {root_url}')
            req = requests.get(url)
            if req.status_code != 200:
                log.info(f'Downloading {filename} failed: {req.status_code}')
                return None
            with open(os.path.join(rel_dataset_path, filename), 'wb') as f:
                f.write(req.content)
            with bz2.open(os.path.join(rel_dataset_path, filename), 'rb') as bz_file:
                with open(os.path.join(rel_dataset_path, txt_filename), 'wb') as out_file:
                    out_file.write(bz_file.read())
            os.remove(os.path.join(rel_dataset_path, filename))

        return os.path.abspath(os.path.join(rel_dataset_path, txt_filename))


def datetimeMsg2TimestampMsg(msg:str):
    msg = msg.split('|')
    msg_date = datetime.datetime.strptime(msg[1], '%Y-%m-%d %H:%M:%S')
    msg[1] = str(int(msg_date.timestamp()))
    return '|'.join(msg)

def read_event_data(events_root, event_name, event_start_time, event_end_time, event_leaker_as):
    event_start_time = datetime.datetime.strptime(event_start_time, '%Y-%m-%d %H:%M')
    event_end_time = datetime.datetime.strptime(event_end_time, '%Y-%m-%d %H:%M') if event_end_time else datetime.datetime(year=2100, month=1, day=1)
    event_start_time = str(int(event_start_time.timestamp()))
    event_end_time = str(int(event_end_time.timestamp()))
    log.info(f'Reading {event_name} BGP Messages')
    event_path = os.path.join(events_root, event_name)
    all_msgs = []
    labels = []
    if os.path.exists(os.path.join(event_path, 'decoded')):
        all_files = []
        for root, dirs, files in os.walk(os.path.join(event_path, 'decoded')):
            for file in files:
                if file.endswith('.txt'):
                    all_files.append(os.path.join(root, file))
        all_files.sort()
        event_happening = False
        for filename in tqdm(all_files):
            with open(filename, 'r') as f:
                for line in f:
                    if line.split('|')[2] != 'A':
                        continue
                    line = datetimeMsg2TimestampMsg(line)
                    all_msgs.append(line.strip())
                    # msg_date = datetime.datetime.strptime(line.split('|')[1], '%Y-%m-%d %H:%M:%S')
                    msg_date = line.split('|')[1]
                    if msg_date>=event_start_time and msg_date<=event_end_time:
                        event_happening = True
                    else:
                        event_happening = False
                    labels.append(1 if event_happening and str(event_leaker_as) in line.split('|')[6] else 0)

    # 均衡样本
    # log.info('Balancing Samples')
    # rate = 1 # 负样本/正样本
    # pos_num = labels.count(1)
    # pos_indices = np.where(np.array(labels) == 1)[0].tolist()
    # neg_indices = np.where(np.array(labels) == 0)[0].tolist()
    # np.random.seed(42)  # 固定随机种子以确保结果可复现
    # sampled_neg_indices = np.random.choice(neg_indices, size=int(rate * pos_num), replace=False)
    # # 合并正负样本下标
    # selected_indices = pos_indices + sampled_neg_indices.tolist()
    # selected_indices.sort()
    # selected_msgs = []
    # for i in tqdm(selected_indices):
    #     selected_msgs.append(all_msgs[i])
    # all_msgs = selected_msgs
    # # all_msgs = [all_msgs[i] for i in selected_indices]
    # # labels = [labels[i] for i in selected_indices]
    # # all_msgs = np.array(all_msgs)[selected_indices].tolist()
    # labels = np.array(labels)[selected_indices].tolist()
    log.info(f'Read Complete')
    return all_msgs, labels

# def AsPathDataClean(hops):
#     # 作者代码里的
#     anomaly=[0,23456,65535]
#     tmphops=copy.copy(hops)
#     for item in hops:
#         if eval(item) in anomaly or (eval(item)<64511 and eval(item)>64496) or (eval(item)<64512 and eval(item)>65534):
#             tmphops.remove(item)
#     #DataClean 1. 去除连续重复的 AS
#     def remove_consecutive_duplicates(as_path):
#         return [as_path[i] for i in range(len(as_path)) if i == 0 or as_path[i] != as_path[i - 1]]
#     tmphops = remove_consecutive_duplicates(tmphops)
#     #DataClean 2. 去除路径中的环路
#     # todo: 是否需要考虑两层以上的环路？
#     def remove_loops(as_path):
#         for i in range(len(as_path) - 2):
#             if as_path[i] == as_path[i+2]:  # 寻找环路（AS[i] == AS[i+2]）
#                 as_path[i+1] = None
#                 as_path[i+2] = None
#         return [asn for asn in as_path if asn is not None]
#     tmphops = remove_loops(tmphops)
    
#     return tmphops

def AsPathDataClean(hops):
    # 去除异常AS号
    anomaly = {0, 23456, 65535}
    tmphops = []
    for item in hops:
        as_num = int(item)  # 使用int替代eval提高效率
        if as_num in anomaly:
            continue
        if 64496 <= as_num <= 65551:  # 过滤保留区间
            continue
        tmphops.append(item)
    # DataClean 1. 去除连续重复的AS
    def remove_consecutive_duplicates(as_path):
        if not as_path:
            return []
        cleaned = [as_path[0]]
        for asn in as_path[1:]:
            if asn != cleaned[-1]:
                cleaned.append(asn)
        return cleaned
    tmphops = remove_consecutive_duplicates(tmphops)
    # DataClean 2. 去除路径中的环路（优化遍历方式）
    def remove_loops(as_path):
        path = as_path.copy()
        i = 0
        while i < len(path) - 2:
            if path[i] == path[i + 2]:
                # 标记i+1和i+2为None
                path[i+1] = path[i+2] = None
                i += 3  # 跳过已处理部分
            else:
                i += 1
        return [asn for asn in path if asn is not None]
    tmphops = remove_loops(tmphops)
    return tmphops


def parse_as_path(as_path:list, graph:nx.DiGraph):
    weight_list = []
    for i in range(len(as_path)-1):
        as1 = int(as_path[i])
        as2 = int(as_path[i + 1])
        if graph.has_edge(as1, as2):
            weight = -graph[as1][as2]['weight'] # 前往provider应当是-1，此时应该取反
        elif graph.has_edge(as2, as1):
            weight = graph[as2][as1]['weight']
        else:
            weight = None
        # 如果没有as关系没有这条边，则跳过
        if weight is None:
            continue
        weight_list.append(weight)
    return weight_list

def expand_as_set(as_path:str):

    """
    展开AS_PATH中的AS_SET，生成所有可能的AS路径组合。
    参数:
        as_path (list): AS_PATH列表，AS_SET用集合表示。
    返回:
        list: 所有可能的AS路径组合。
    """
    from itertools import product
    # 将AS_PATH中的每个元素转换为列表（如果是AS_SET，则展开为多个AS）
    expanded_paths = []
    for as_segment in as_path.split(' '):
        if as_segment.startswith('{'):  # 如果是AS_SET
            expanded_paths.append(list(as_segment[1:-1].split(',')))  # 将AS_SET转换为列表
        else:  # 如果是单个AS
            expanded_paths.append([as_segment])  # 将单个AS放入列表

    # 使用笛卡尔积生成所有可能的AS路径组合
    all_paths = list(product(*expanded_paths))
    all_paths = [list(x) for x in all_paths]
    return all_paths


def varify_path(as_path:list, weight_list:list[int],):
    """验证是否合法路径

    Args:
        weight_list (list[int]): as路径权重

    Returns:
        flag: 是否为合法路径
    """
    flag = True
    anomaly_tuples = set()
    for i in range(1, len(weight_list)):
        if weight_list[i] < weight_list[i - 1]:
            flag = False
            anomaly_tuples.add((as_path[i], as_path[i+1]))
            # break
        if weight_list[i]==0 and weight_list[i-1]==0:
            flag = False
            anomaly_tuples.add((as_path[i], as_path[i+1]))
            # break
    return flag, anomaly_tuples


def ValleyFree(data_source):
    # with open('events.json', 'r') as f:
    #     events = json.load(f)
    events = read_event_list(evnet_type='leak')
    all_event_results = [] # 取全部异常as判断的结果
    top10_event_results = [] # 只取前十个异常as判断的结果
    for event in events:
        event_name = event['event_name']
        event_start_time = event['event_start_time']
        event_end_time = event['event_end_time']
        event_leaker_as = event['leaker_as']
    # for event_name in sorted(os.listdir(events_root)):
        event_path = os.path.join(events_root, event_name)
        # if event_name.startswith('.') or not os.path.isdir(event_path):
        #     # 排除macos下的.DS_Store文件
        #     continue
        # if event_name != 'leak-20160422-AWS_Route_Leak':
        #     continue
        log.info(f'Processing event: {event_name}')
        save_dir = os.path.join('mydata_test_result', event_name)
        os.makedirs(save_dir, exist_ok=True)
        # if os.path.exists(os.path.join(save_dir, 'metrics.txt')):
        #     os.rename(os.path.join(save_dir, 'metrics.txt'), os.path.join(save_dir, 'metrics_old.txt'))
        #     continue
        # if os.path.exists(os.path.join(save_dir, 'pred_labels.txt')) and os.path.exists(os.path.join(save_dir, 'true_labels.txt')):
        #     with open(os.path.join(save_dir, 'pred_labels.txt'), 'r') as f:
        #         pred_y = eval(f.read())
        #     with open(os.path.join(save_dir, 'true_labels.txt'), 'r') as f:
        #         true_labels = eval(f.read())
        #     metric = Metric(event_name)
        #     metric.calculate_metrics_point_wise(true_labels, pred_y)
        #     continue
        all_msgs, true_labels = read_event_data(events_root, event_name, event_start_time, event_end_time, event_leaker_as)
        # event_date = event_name.split('-')[1]
        # event_date = datetime.datetime.strptime(event_date, '%Y%m%d')
        event_start_time = datetime.datetime.strptime(event['event_start_time'], '%Y-%m-%d %H:%M') 
        # event_end_time = datetime.datetime.strptime(event['event_end_time'], '%Y-%m-%d %H:%M') 
        rel_filepath = get_rel_filepath(source=data_source, event_start_date=event_start_time)
        if event_start_time.replace(day=1)<datetime.datetime(year=2015, month=12, day=1):
            G = nx.read_edgelist(rel_filepath, nodetype=int, data=(("weight", int),), comments='#', delimiter="|", create_using=nx.DiGraph)
        else:
            G = nx.read_edgelist(rel_filepath, nodetype=int, data=(("weight", int),("source", str),), comments='#', delimiter="|", create_using=nx.DiGraph)
        
        AnomalyTupleDict = {}
        pred_y = []
        for msg in tqdm(all_msgs):
            msg = msg.split('|')
            if msg[2] != 'A':
                continue
            as_paths = msg[6]
            as_paths = expand_as_set(as_paths)
            is_valid = True
            for as_path in as_paths:
                as_path = AsPathDataClean(as_path)
                weight_list = parse_as_path(as_path=as_path, graph=G)
                is_valid, anomaly_tuples = varify_path(as_path, weight_list)
                if not is_valid:
                    for t in anomaly_tuples:
                        AnomalyTupleDict.setdefault(t, []).append([msg[6], msg[1]])
                    # break
            pred_y.append(0 if is_valid else 1)
        metric = Metric(event_name)
        metric.calculate_metrics(true_labels, pred_y)

        AnomalyTupleDict = dict(sorted(AnomalyTupleDict.items(), key=lambda x:(len(x[1]), x[0]), reverse=True))
        anomaly_asn = set()
        s=''
        for k, v in list(AnomalyTupleDict.items())[:10]:
            s+=f'triplet:{k}, as_path_num:{len(v)}\n'
            print(f'triplet:{k}, as_path_num:{len(v)}')
            # print(f'leaker:{k[0]}, leaked_to:{k[1]}, as_path_num:{len(v)}')
            anomaly_asn.add(k[0])
            anomaly_asn.add(k[1])
        with open(os.path.join(save_dir, 'anomalys-all_triplets.txt'), 'w') as f:
            f.write(s)
        if str(event_leaker_as) in anomaly_asn:
            top10_event_results.append(1)
        else:
            top10_event_results.append(0)

        # event-wise计算所有的告警
        for k, v in AnomalyTupleDict.items():
            anomaly_asn.add(k[0])
            anomaly_asn.add(k[1])

        if str(event_leaker_as) in anomaly_asn:
            all_event_results.append(1)
        else:
            all_event_results.append(0)
    print(f'Event-Wise: {top10_event_results.count(1)}/{len(top10_event_results)}')
    print(f'loose Event-Wise: {all_event_results.count(1)}/{len(all_event_results)}')
    with open(os.path.join('mydata_test_result', 'event-wise metric.txt'), 'w') as f:
        f.write(str(top10_event_results))
        f.write(f'\nEvent-Wise: {top10_event_results.count(1)}/{len(top10_event_results)}')
    with open(os.path.join('mydata_test_result', 'event-wise metric(all anomaly).txt'), 'w') as f:
        f.write(str(all_event_results))
        f.write(f'\nloose Event-Wise: {all_event_results.count(1)}/{len(all_event_results)}')


def calc_point_wise_metric():
    pred_labels = []
    true_labels = []
    result_path = "mydata_test_result"
    for event_name in tqdm(sorted(os.listdir(result_path))):
        event_path = os.path.join(result_path, event_name)
        if not os.path.isdir(event_path):
            continue
        with open(os.path.join(event_path, 'pred_labels.txt')) as f:
            # pred_labels.extend(eval(f.read()))
            pred_labels.extend(json.load(f))
        with open(os.path.join(event_path, 'true_labels.txt')) as f:
            # true_labels.extend(eval(f.read()))
            true_labels.extend(json.load(f))
    (TN, FP, FN, TP), report, precision, recall, F1_score = Metric._calc_metrics(true_labels, pred_labels)
    s = f"TN={TN}, FP={FP}, FN={FN}, TP={TP}\nprecision={precision}, recall={recall}, F1-Score={F1_score}\n\n{report}"
    print(s)
    with open(os.path.join(result_path, 'point-wise metrics.txt'), 'w') as f:
        f.write(s)


def eval_on_roll_dataset():
    dataset_name = 'allsample.csv'
    all_data = []
    with open(dataset_name, 'r') as f:
        all_data = f.read().strip().split('\n')
    all_data = [data.split(',') for data in all_data]
    all_data = [[int(x[0]), x[1:4], int(x[-1])] for x in all_data]
    all_data.sort(key=lambda x: x[0])
    year=2021
    G = None
    true_y = []
    pred_y = []
    for data in all_data:
        if data[0] != year or G is None:
            year = data[0]
            rel_filepath = get_rel_filepath(source='CAIDA', event_start_date=datetime.datetime(year=year, month=1, day=1))
            G = nx.read_edgelist(rel_filepath, nodetype=int, data=(("weight", int),("source", str),), comments='#', delimiter="|", create_using=nx.DiGraph)
        weight_list = parse_as_path(as_path=data[1], graph=G)
        pred_label = varify_path(weight_list)
        true_label = data[-1]
        pred_y.append(0 if pred_label else 1)
        true_y.append(true_label)
    (TN, FP, FN, TP), report, precision, recall, F1_score = Metric._calc_metrics(true_y, pred_y)
    s = f"TN={TN}, FP={FP}, FN={FN}, TP={TP}\nprecision={precision}, recall={recall}, F1-Score={F1_score}\n\n{report}"
    print(s)


if __name__ == '__main__':
    # eval_on_roll_dataset()
    ValleyFree(data_source='CAIDA')
    calc_point_wise_metric()