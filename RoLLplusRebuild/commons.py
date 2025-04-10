import bz2
import datetime
import gzip
import json
import os
import logging
import re
import threading
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm # py3.9后是系统库
from ASFeature import ASFeature
def beijing(sec, what):
    beijing_time = datetime.datetime.now(ZoneInfo('Asia/Shanghai')) # 返回北京时间
    return beijing_time.timetuple()
logging.Formatter.converter = beijing
logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S', format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

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
                    # line = datetimeMsg2TimestampMsg(line)
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


featureNames=[
    'asDistance0', 'asDistance1', 'asDistance2',
    'asDegree0','asDegree1','asDegree2',
    'asAddressSpace0','asAddressSpace1','asAddressSpace2',
    'asCountry','asRIR',
    'asType0','asType1','asType2',
    'asBetweennessCentrality0','asBetweennessCentrality1','asBetweennessCentrality2',
    'asClosenessCentrality0', 'asClosenessCentrality1', 'asClosenessCentrality2',
    'asEigenvectorCentrality0', 'asEigenvectorCentrality1', 'asEigenvectorCentrality2',
    'asClusteringCoefficient0','asClusteringCoefficient1','asClusteringCoefficient2',
    'asSquareClustering0', 'asSquareClustering1', 'asSquareClustering2',
    # 'asRouterNumber0','asRouterNumber1','asRouterNumber2',
    "asAverageNeighborDegree0", "asAverageNeighborDegree1", "asAverageNeighborDegree2",
    # "asMaxCliqueSize0", "asMaxCliqueSize1", "asMaxCliqueSize2",
    "asTrianglesClustering0", "asTrianglesClustering1", "asTrianglesClustering2",
    # 'ixp',
    'label'
]

def getModel_my()->RandomForestClassifier:
    
    pima = pd.read_csv("reGenerateSamples.csv", header=None, usecols=[
        4,5,6,
        7,8,9,
        10,11,12,
        13,14,
        15,16,17,
        18,19,20,
        21,22,23,
        24,25,26,
        27,28,29,
        30,31,32,
        33,34,35,
        36,37,38,
        39
        # 39,40,41,
        # 42,43,44,
        # 45,
        
        ],names=featureNames)
    X = pima[featureNames[0:-1]] # Features
    y = pima.label # Target variable
    clf=RandomForestClassifier(n_estimators=100, max_depth=10,oob_score=True)
    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X,y)
    return clf


def detectTriplets(triplets:set, ASFeatures:ASFeature, TripletsCache:dict, Classifier:RandomForestClassifier):
    global detect_time_consume
    valid_triplets_2_label = {} # 合法的，数据库有as的三元组，以及对应的label
    tmp_triplets = [] # 临时存放待检测的三元组， 与X一一对应
    X = [] # 缓存中没有的三元组的feature，用来检测
    for triplet in triplets:
        flag = TripletsCache.get(triplet, -1)
        if flag!= -1:
            valid_triplets_2_label[triplet] = flag
            continue
        # 排除sibling as，直接设为正常
        for siblings in ASFeatures.SiblingAs:
            if len(set(siblings).intersection(set(triplet)))>1:
                TripletsCache[triplet] = 0
                valid_triplets_2_label[triplet] = 0
                continue
        asDistance0 = ASFeatures.getASDistance(triplet[0])
        asDistance1 = ASFeatures.getASDistance(triplet[1])
        asDistance2 = ASFeatures.getASDistance(triplet[2])
        asDegree0 = ASFeatures.getASDegree(triplet[0])
        asDegree1 = ASFeatures.getASDegree(triplet[1])
        asDegree2 = ASFeatures.getASDegree(triplet[2])
        asAddressSpace0 = ASFeatures.getASAddressSpace(triplet[0])
        asAddressSpace1 = ASFeatures.getASAddressSpace(triplet[1])
        asAddressSpace2 = ASFeatures.getASAddressSpace(triplet[2])
        asCountry = ASFeatures.getTripletCountry(triplet)
        asRIR = ASFeatures.getTripletRIR(triplet)
        asType0 = ASFeatures.getASType(triplet[0])
        asType1 = ASFeatures.getASType(triplet[1])
        asType2 = ASFeatures.getASType(triplet[2])
        asBetweennessCentrality0 = ASFeatures.getASBetweennessCentrality(triplet[0])
        asBetweennessCentrality1 = ASFeatures.getASBetweennessCentrality(triplet[1])
        asBetweennessCentrality2 = ASFeatures.getASBetweennessCentrality(triplet[2])
        asClosenessCentrality0 = ASFeatures.getASClosenessCentrality(triplet[0])
        asClosenessCentrality1 = ASFeatures.getASClosenessCentrality(triplet[1])
        asClosenessCentrality2 = ASFeatures.getASClosenessCentrality(triplet[2])
        asEigenvectorCentrality0 = ASFeatures.getASEigenvectorCentrality(triplet[0])
        asEigenvectorCentrality1 = ASFeatures.getASEigenvectorCentrality(triplet[1])
        asEigenvectorCentrality2 = ASFeatures.getASEigenvectorCentrality(triplet[2])
        asClusteringCoefficient0 = ASFeatures.getASClusteringCoefficient(triplet[0])
        asClusteringCoefficient1 = ASFeatures.getASClusteringCoefficient(triplet[1])
        asClusteringCoefficient2 = ASFeatures.getASClusteringCoefficient(triplet[2])
        asSquareClustering0 = ASFeatures.getASSquareClustering(triplet[0])
        asSquareClustering1 = ASFeatures.getASSquareClustering(triplet[1])
        asSquareClustering2 = ASFeatures.getASSquareClustering(triplet[2])
        # asRouterNumber0 = ASFeatures.getASRouterNumber(triplet[0])
        # asRouterNumber1 = ASFeatures.getASRouterNumber(triplet[1])
        # asRouterNumber2 = ASFeatures.getASRouterNumber(triplet[2])
        asAverageNeighborDegree0 = ASFeatures.getASAverageNeighborDegree(triplet[0])
        asAverageNeighborDegree1 = ASFeatures.getASAverageNeighborDegree(triplet[1])
        asAverageNeighborDegree2 = ASFeatures.getASAverageNeighborDegree(triplet[2])
        # asMaxCliqueSize0 = ASFeatures.getASMaxCliqueSize(triplet[0])
        # asMaxCliqueSize1 = ASFeatures.getASMaxCliqueSize(triplet[1])
        # asMaxCliqueSize2 = ASFeatures.getASMaxCliqueSize(triplet[2])
        asTrianglesClustering0 = ASFeatures.getASTrianglesClustering(triplet[0])
        asTrianglesClustering1 = ASFeatures.getASTrianglesClustering(triplet[1])
        asTrianglesClustering2 = ASFeatures.getASTrianglesClustering(triplet[2])
        a=[
            asDistance0, asDistance1, asDistance2,
            asDegree0, asDegree1, asDegree2,
            asAddressSpace0, asAddressSpace1, asAddressSpace2,
            asCountry, asRIR,
            asType0, asType1, asType2,
            asBetweennessCentrality0, asBetweennessCentrality1, asBetweennessCentrality2,
            asClosenessCentrality0, asClosenessCentrality1, asClosenessCentrality2,
            asEigenvectorCentrality0, asEigenvectorCentrality1, asEigenvectorCentrality2,
            asClusteringCoefficient0, asClusteringCoefficient1, asClusteringCoefficient2,
            asSquareClustering0, asSquareClustering1, asSquareClustering2,
            # asRouterNumber0, asRouterNumber1, asRouterNumber2,
            asAverageNeighborDegree0, asAverageNeighborDegree1, asAverageNeighborDegree2,
            # asMaxCliqueSize0, asMaxCliqueSize1, asMaxCliqueSize2,
            asTrianglesClustering0, asTrianglesClustering1, asTrianglesClustering2,
        ]

        # a = [str(x) for x in a]
        # s=','.join(a)+'\n'
        if -1000 in a:
            s = f"triplet: {str(triplet)}"
            # if -1000 in a[11:14]: # 排除astype不全的情况
            #     continue
            for idx, feature in enumerate(a):
                if feature == -1000:
                    s += f"\n{featureNames[idx]}: {feature}"
            log.error(s)
            # triplets.remove(triplet)
            continue
        tmp_triplets.append(triplet)
        X.append(a)
    if not X:
        # 默认没有异常
        return valid_triplets_2_label
    # X = np.array(X)
    X = pd.DataFrame(X, columns=featureNames[0:-1])
    Y = Classifier.predict(X)
    for triplet, label in zip(tmp_triplets, Y):
        TripletsCache[triplet] = int(label)
        valid_triplets_2_label[triplet] = int(label)
    return valid_triplets_2_label

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

def decodeMRT(filepaths:list):
    if not isinstance(filepaths, list):
        filepaths = [filepaths]
    all_data = []
    log.info("decoding MRT files......")
    for filepath in tqdm(filepaths):
        decoded_data = os.popen(f"bgpdump -q -m {filepath}").read()
        decoded_data = decoded_data.strip('\n').split("\n")
        all_data.extend(decoded_data)
    return all_data

def get_rel_filepath(rel_file_root, source, event_start_date:datetime.datetime):
    rel_dataset_path = os.path.join(rel_file_root, source)
    os.makedirs(rel_dataset_path, exist_ok=True)
    if source == "CAIDA":
        root_url = "https://publicdata.caida.org/datasets/as-relationships"
        date1 = event_start_date.replace(day=1)
        if date1<datetime.datetime(year=2015, month=12, day=1):
            filename = f'{date1.strftime("%Y%m%d")}.as-rel.txt.bz2'
            url = f'{root_url}/serial-1/{filename}'
        else:
            filename = f'{date1.strftime("%Y%m%d")}.as-rel2.txt.bz2'
            url = f'{root_url}/serial-2/{filename}'
        txt_filename = filename.removesuffix('.bz2')
        if not os.path.exists(os.path.join(rel_dataset_path, txt_filename)):
            log.info(f'Downloading {filename} from {url}')
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
    
def get_org_filepath(org_file_root, source, event_start_date, use_local=False, use_latest=False):
    """获取org数据集

    Args:
        source (str): 来源(CAIDA)
        event_start_date (datetime): 事件开始时间
        use_local (bool, optional): 是否直接使用本地数据而非从网页获取最接近的数据. Defaults to False.
        use_latest (bool, optional): 是否使用最近日期的org数据而非最接近日期的数据. Defaults to False.
    """
    def find_closest_file(file_names, given_date):
        closest_file = None
        min_diff = float('inf')  # 初始化最小差值为无穷大
        for file_name in file_names:
            try:
                file_date = datetime.datetime.strptime(file_name[:8], "%Y%m%d")
            except ValueError:
                continue  # 如果日期无效，跳过该文件
            diff = abs((file_date - given_date).days)
            if diff < min_diff:
                min_diff = diff
                closest_file = file_name
        return closest_file
    org_dataset_path = os.path.join(org_file_root, source)
    os.makedirs(org_dataset_path, exist_ok=True)
    if source == "CAIDA":
        root_url = "https://publicdata.caida.org/datasets/as-organizations"
        if not use_local or not os.listdir(org_dataset_path):
            # 获取网页上所有的org文件，查找最合适的一个
            page = requests.get(root_url)
            if page.status_code != 200:
                log.info(f"request file list faild: {page.status_code}")
            page = page.text
            pattern = r'\d{8}\.as-org2info\.txt\.gz'
            matches = re.findall(pattern, page)
        else:
            matches = os.listdir(os.path.join(org_file_root, source))

        if not use_latest or not os.listdir(org_dataset_path):
            filename = find_closest_file(matches, event_start_date)
        else:
            filename = sorted(os.listdir(org_dataset_path))[-1]  # 最新的版本
        url = f'{root_url}/{filename}'
        txt_filename = filename.removesuffix('.gz')
        if not os.path.exists(os.path.join(org_dataset_path, txt_filename)):
            log.info(f'Downloading {filename} from {url}')
            req = requests.get(url)
            if req.status_code != 200:
                log.info(f'Downloading {filename} failed: {req.status_code}')
                return None
            with open(os.path.join(org_dataset_path, filename), 'wb') as f:
                f.write(req.content)
            g_file = gzip.GzipFile(os.path.join(org_dataset_path, filename))
            open(os.path.join(org_dataset_path, txt_filename), "wb+").write(g_file.read())
            # with bz2.open(os.path.join(org_dataset_path, filename), 'rb') as bz_file:
            #     with open(os.path.join(org_dataset_path, txt_filename), 'wb') as out_file:
            #         out_file.write(bz_file.read())
            os.remove(os.path.join(org_dataset_path, filename))

        return os.path.abspath(os.path.join(org_dataset_path, txt_filename))


def get_astype_filepath(astype_file_root, event_start_date, use_local=False, use_latest=True):
    """获取astype数据集

    Args:
        source (str): 来源(CAIDA)
        event_start_date (datetime): 事件开始时间
        use_local (bool, optional): 是否直接使用本地数据而非从网页获取最接近的数据. Defaults to False.
        use_latest (bool, optional): 是否使用最新的astype数据(2021.4). Defaults to True.
    """
    def find_closest_file(file_names, given_date):
        closest_file = None
        min_diff = float('inf')  # 初始化最小差值为无穷大
        for file_name in file_names:
            try:
                file_date = datetime.datetime.strptime(file_name[:8], "%Y%m%d")
            except ValueError:
                continue  # 如果日期无效，跳过该文件
            diff = abs((file_date - given_date).days)
            if diff < min_diff:
                min_diff = diff
                closest_file = file_name
        return closest_file

    os.makedirs(astype_file_root, exist_ok=True)

    root_url = "https://publicdata.caida.org/datasets/as-classification_restricted"
    if not use_local:
        # 获取网页上所有的org文件，查找最合适的一个
        page = requests.get(root_url)
        if page.status_code != 200:
            log.info(f"request file list faild: {page.status_code}")
        page = page.text
        pattern = r'\d{8}\.as2types\.txt\.gz'
        matches = re.findall(pattern, page)
    else:
        matches = os.listdir(astype_file_root)
    if not use_latest:
        filename = find_closest_file(matches, event_start_date)
    else:
        filename = '20210401.as2types.txt.gz'  # 2021.4 release
    url = f'{root_url}/{filename}'
    txt_filename = filename.removesuffix('.gz')
    if not os.path.exists(os.path.join(astype_file_root, txt_filename)):
        log.info(f'Downloading {filename} from {url}')
        req = requests.get(url)
        if req.status_code != 200:
            log.info(f'Downloading {filename} failed: {req.status_code}')
            return None
        with open(os.path.join(astype_file_root, filename), 'wb') as f:
            f.write(req.content)
        g_file = gzip.GzipFile(os.path.join(astype_file_root, filename))
        open(os.path.join(astype_file_root, txt_filename), "wb+").write(g_file.read())
        # with bz2.open(os.path.join(org_dataset_path, filename), 'rb') as bz_file:
        #     with open(os.path.join(org_dataset_path, txt_filename), 'wb') as out_file:
        #         out_file.write(bz_file.read())
        os.remove(os.path.join(astype_file_root, filename))

    return os.path.abspath(os.path.join(astype_file_root, txt_filename))


def get_pfx2as_filepath(prefix2as_file_root, source, event_start_date:datetime.datetime, use_local=False):
    """获取pfx2as数据集

    Args:
        source (str): 来源(CAIDA)
        event_start_date (datetime): 事件开始时间
        use_local (bool, optional): 是否直接使用本地数据而非从网页获取最接近的数据. Defaults to False.
    """
    def find_closest_file(file_names:list[str], given_date):
        closest_file = None
        min_diff = float('inf')  # 初始化最小差值为无穷大
        for file_name in file_names:
            try:
                file_date = datetime.datetime.strptime(file_name.split('.')[0][-13:], "%Y%m%d-%H%M")
            except ValueError:
                continue  # 如果日期无效，跳过该文件
            diff = abs((file_date - given_date).days)
            if diff < min_diff:
                min_diff = diff
                closest_file = file_name
        return closest_file
    pfx2as_dataset_path = os.path.join(prefix2as_file_root, source)
    os.makedirs(pfx2as_dataset_path, exist_ok=True)
    if source == "CAIDA":
        root_url = "https://publicdata.caida.org/datasets/routing/routeviews-prefix2as"
        if not use_local:
            _year = event_start_date.year
            _month = event_start_date.month
            if event_start_date.year<2005:
                _year=2005
            # 获取网页上的年份
            page = requests.get(f'{root_url}/{_year}')
            if page.status_code != 200:
                log.info(f"request file list from {root_url}/{_year} faild: {page.status_code}")
            page = page.text
            months = re.findall(r'href="(\d{2})/"', page)
            if event_start_date.year<2005:
                _month = sorted(months)[0]
            else:
                _month = min(sorted(months, reverse=True), key=lambda x: abs(int(x) - event_start_date.month))
            page = requests.get(f'{root_url}/{_year}/{_month}')
            if page.status_code != 200:
                log.info(f"request file list from {root_url}/{_year}/{_month} faild: {page.status_code}")
            page = page.text
            pattern = r'href="(routeviews-\w+-\d{8}-\d{4}\.pfx2as\.gz)'
            matches = re.findall(pattern, page)
            root_url = f'{root_url}/{_year}/{_month}'
        else:
            matches = os.listdir(os.path.join(prefix2as_file_root, source))
        filename = find_closest_file(matches, event_start_date)
        url = f'{root_url}/{filename}'
        txt_filename = filename.removesuffix('.gz')
        if not os.path.exists(os.path.join(pfx2as_dataset_path, txt_filename)):
            log.info(f'Downloading {filename} from {url}')
            req = requests.get(url)
            if req.status_code != 200:
                log.info(f'Downloading {filename} failed: {req.status_code}')
                return None
            with open(os.path.join(pfx2as_dataset_path, filename), 'wb') as f:
                f.write(req.content)
            g_file = gzip.GzipFile(os.path.join(pfx2as_dataset_path, filename))
            open(os.path.join(pfx2as_dataset_path, txt_filename), "wb+").write(g_file.read())
            os.remove(os.path.join(pfx2as_dataset_path, filename))
        return os.path.abspath(os.path.join(pfx2as_dataset_path, txt_filename))
    elif source == "RIPE":
        file_url = "https://www.ris.ripe.net/dumps/riswhoisdump.IPv4.gz"
        txt_filename = f"{datetime.datetime.now().strftime('%Y%m%d')}.riswhoisdump.IPv4"
        log.info(f'Downloading {txt_filename} from {file_url}')
        req = requests.get(file_url)
        if req.status_code != 200:
            log.info(f'Downloading riswhoisdump.IPv4.gz failed: {req.status_code}')
            return None
        with open(os.path.join(pfx2as_dataset_path, 'riswhoisdump.IPv4.gz'), 'wb') as f:
            f.write(req.content)
        g_file = gzip.GzipFile(os.path.join(pfx2as_dataset_path, 'riswhoisdump.IPv4.gz'))
        open(os.path.join(pfx2as_dataset_path, txt_filename), "wb+").write(g_file.read())
        os.remove(os.path.join(pfx2as_dataset_path, 'riswhoisdump.IPv4.gz'))
        return os.path.abspath(os.path.join(pfx2as_dataset_path, txt_filename))


class FileDownloadTimeDatabase:
    def __init__(self, filepath):
        self.filepath = filepath
        self.lock = threading.Lock()

    def _get_data(self)->dict:
        if os.path.exists(self.filepath):
            with open(self.filepath, 'r') as f:
                data = json.load(f)
        else:
            data = {}
        return data
    
    def _save_data(self, data):
        with open(self.filepath, 'w') as f:
            json.dump(data, f, indent=4)

    def updateKey(self, key, update_time:datetime.datetime):
        with self.lock:
            data = self._get_data()
            data[key] = update_time.strftime("%Y-%m-%d %H-%M-%S")
            self._save_data(data)
    
    def getKey(self, key):
        with self.lock:
            data = self._get_data()
            return data[key] if key in data.keys() else None

    def hasKey(self, key):
        with self.lock:
            data = self._get_data()
            if key in data.keys():
                return True
            else:
                return False