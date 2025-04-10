import cProfile
import copy
import json
import os
import datetime
import re
import time
import numpy as np
import pandas as pd
import requests
from commons import Metric, read_event_data, AsPathDataClean, getModel_my,\
        detectTriplets, expand_as_set, read_event_list
from alert_compress import merge_anomaly_msgs
from tqdm import tqdm
import bz2, gzip

from sklearn.ensemble import RandomForestClassifier
from ASFeature import ASFeature
import networkx as nx
import logging
from zoneinfo import ZoneInfo # py3.9后是系统库
def beijing(sec, what):
    beijing_time = datetime.datetime.now(ZoneInfo('Asia/Shanghai')) # 返回北京时间
    return beijing_time.timetuple()
logging.Formatter.converter = beijing
logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S', format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

rel_file_root = './mydata_rel_data'
org_file_root = './mydata_org_data'
astype_file_root = './mydata_astype_data'
prefix2as_file_root = './mydata_prefix2as_data'
# events_root = './events'
events_root = '/data/data/anomaly-event-routedata'

TripletsCache = {}
total_time = time.time()
detect_time_consume = 0


def get_rel_filepath(source, event_start_date:datetime.datetime):
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
    
def get_org_filepath(source, event_start_date, use_local=False, use_latest=False):
    """获取org数据集

    Args:
        source (str): 来源(CAIDA)
        event_start_date (datetime): 事件开始时间
        use_local (bool, optional): 是否直接使用本地数据而非从网页获取最接近的数据. Defaults to False.
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
        if not use_local:
            # 获取网页上所有的org文件，查找最合适的一个
            page = requests.get(root_url)
            if page.status_code != 200:
                log.info(f"request file list faild: {page.status_code}")
            page = page.text
            pattern = r'\d{8}\.as-org2info\.txt\.gz'
            matches = re.findall(pattern, page)
        else:
            matches = os.listdir(os.path.join(org_file_root, source))

        if not use_latest:
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


def get_astype_filepath(event_start_date, use_local=False, use_latest=True):
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


def get_pfx2as_filepath(source, event_start_date:datetime.datetime, use_local=False):
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


Classifier = getModel_my()

def eval_mydata_optimized(data_source):
    # 优化性能，一次检测多条消息
    global total_time,  detect_time_consume
    unseen_asn = set()
    events_info = []
    # with open('events.json', 'r') as f:
    #     events_info = json.load(f)
    events_info = read_event_list(evnet_type='leak')
    all_event_results = [] # 取全部异常as判断的结果
    top10_event_results = [] # 只取前十个异常as判断的结果
    for event in events_info:
        anomaly_msgs = []
        pred_y = []

        event_name = event['event_name']
        event_start_time = event['event_start_time']
        event_end_time = event['event_end_time']
        event_leaker_as = event['leaker_as']
        # event_path = os.path.join(events_root, event_name)
        # if event_name.startswith('.') or not os.path.isdir(event_path):
        #     # 排除macos下的.DS_Store文件
        #     continue
        # if event_name != 'leak-20190606-SafeHost_leak':
        #     continue
        log.info(f'Processing event: {event_name}')
        save_dir = os.path.join('mydata_test_result', event_name)
        # if os.path.exists(save_dir) and os.path.exists(os.path.join(save_dir, 'anomalys-all_triplets.txt')):
        #     continue
        # if os.path.exists(os.path.join(save_dir, 'anomalys-all_triplets.txt')):
        #     os.remove(os.path.join(save_dir, 'anomalys-all_triplets.txt'))
        # with open(os.path.join(save_dir, 'true_labels.txt'), 'r') as f:
        #     true_labels = eval(f.read())
        # with open(os.path.join(save_dir, 'pred_labels.txt'), 'r') as f:
        #     pred_y = eval(f.read())
        os.makedirs(save_dir, exist_ok=True)
        all_msgs, true_labels = read_event_data(events_root, event_name, event_start_time, event_end_time, event_leaker_as)
        
        # metric = Metric(event_name)
        # metric.calculate_metrics_point_wise(true_labels, pred_y)
        # event_date = event_name.split('-')[1]
        # event_date = datetime.datetime.strptime(event_date, '%Y%m%d')
        event_start_time = datetime.datetime.strptime(event_start_time, '%Y-%m-%d %H:%M')
        rel_filepath = get_rel_filepath(source=data_source, event_start_date=event_start_time)
        org_filepath = get_org_filepath(source=data_source, event_start_date=event_start_time, use_local=True, use_latest=True) # 直接使用最新的org数据
        pfx2as_filepath = get_pfx2as_filepath(source="CAIDA", event_start_date=event_start_time, use_local=True)
        asTopology = ASFeature(rel_filename=rel_filepath, org_filename=org_filepath, pfx2as_filename=pfx2as_filepath)
        AStriplets = set()
        batch_triplets = {} # 一批次的消息，每条消息对应的三元组
        cache_hit_cnt=0
        asn_unseen_skip_cnt=0
        total_time = time.time()
        detect_time_consume = 0
        for msg_idx, msg in enumerate(tqdm(all_msgs)):
            batch_triplets.setdefault(msg_idx, set())
            msg = msg.split('|')
            asn = msg[4]
            prefix = msg[5]
            as_path = msg[6]
            expanded_aspath = expand_as_set(as_path) # as set路由聚合后将其分离
            for hops in expanded_aspath:
                # hops=as_path.split(" ")
                # 数据清理
                hops=AsPathDataClean(hops)
                # 过滤不存在asrel文件中的asn
                for hop in hops:
                    if int(hop) not in asTopology.asSet and hop not in unseen_asn:
                        unseen_asn.add(hop)
                        # log.error(f"ASN {hop} not found in as rel file")
                    elif hop not in asTopology.as2Prefix4.keys() and hop not in unseen_asn:
                        # and hop not in ASFeatures.as2Prefix6.keys():
                        unseen_asn.add(hop)
                        # log.error(f"ASN {hop} not found in pfx2as file")
                    elif hop not in asTopology.as2OrgDict.keys() and hop not in unseen_asn:
                        unseen_asn.add(hop)
                        # log.error(f"ASN {hop} not found in as org file")
                    elif hop not in asTopology.asType.keys() and hop not in unseen_asn:
                        unseen_asn.add(hop)
                        # log.error(f"ASN {hop} not found in as Type file")
                if len(hops) > 2:
                    for i in range(1,len(hops)-1):
                        if hops[i-1] in unseen_asn or hops[i] in unseen_asn or hops[i+1] in unseen_asn:
                            continue
                        t = (hops[i-1],hops[i],hops[i+1])
                        batch_triplets.setdefault(msg_idx, set()).add(t)
                        AStriplets.add(t)

            if len(AStriplets)<100 and msg_idx!=len(all_msgs)-1:
                continue
            valid_triplets_2_label = detectTriplets(AStriplets, asTopology, TripletsCache, Classifier)
            for idx, triplets in batch_triplets.items():
                flag = 0
                anomaly_triplets = set()
                for t in triplets:
                    if valid_triplets_2_label.get(t, 0)==1:
                        flag = 1
                        anomaly_triplets.add(t)
                pred_y.append(flag)
                if flag:
                    # anomaly_msgs.append([(flag), (true_labels[idx]), triplets, all_msgs[idx]]) # 添加所有三元组而非仅仅是异常的
                    anomaly_msgs.append([(flag), (true_labels[idx]), anomaly_triplets, all_msgs[idx]])
                    # triplet_str = [f'({t[0]},{t[1]},{t[2]})' for t in anomaly_triplets]
                    # anomaly_msgs.append(f'pred:{flag} actual:{true_labels[idx]}|{",".join(triplet_str)}|{all_msgs[idx]}\n')
            AStriplets = set()
            batch_triplets = {}
        # if not os.path.exists(os.path.join(save_dir, 'metrics.txt')):
        metric = Metric(event_name)
        metric.calculate_metrics(true_labels, pred_y)
        # with open(os.path.join(save_dir, 'anomaly_messages.txt'), 'w') as f:
        #     f.writelines(anomaly_msgs)
        # 告警压缩
        anomaly_tuples = merge_anomaly_msgs(anomaly_msgs) # 异常元组(leaker, leaked_to)
        anomaly_asn = set()
        s=''
        for k, v in list(anomaly_tuples.items())[:10]:
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
        for k, v in anomaly_tuples.items():
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

from concurrent.futures import ProcessPoolExecutor

def ASFeature_wrap(event_name, rel_filepath, org_filepath, pfx2as_filepath):
        log.info(f'Processing event features: {event_name}')
        ASFeature(rel_filename=rel_filepath, org_filename=org_filepath, pfx2as_filename=pfx2as_filepath)
        log.info(f'Event features Done: {event_name}')

def preCalcASFeature_mp(data_source):
    with ProcessPoolExecutor(max_workers=8) as executor:  # 设置最大进程数
        futures = []
        for event_name in sorted(os.listdir(events_root)):
            event_path = os.path.join(events_root, event_name)
            if event_name.startswith('.') or not os.path.isdir(event_path):
                # 排除macos下的.DS_Store文件
                continue

            event_date = event_name.split('-')[1]
            event_date = datetime.datetime.strptime(event_date, '%Y%m%d')
            rel_filepath = get_rel_filepath(source=data_source, event_start_date=event_date)
            org_filepath = get_org_filepath(source=data_source, event_start_date=event_date, use_local=True)
            pfx2as_filepath = get_pfx2as_filepath(source="CAIDA", event_start_date=event_date, use_local=True)
            asTopology = ASFeature(rel_filename=rel_filepath, org_filename=org_filepath, pfx2as_filename=pfx2as_filepath)
            futures.append(executor.submit(ASFeature_wrap, event_name, rel_filepath, org_filepath, pfx2as_filepath))

        results = []
        for future in futures:
            result = future.result()  # 获取每个进程的结果
            if result is not None:
                results.append(result)
    
    return results  # 返回所有计算结果


def preCalcASFeature(data_source):
    with open('events.json', 'r') as f:
        events_info = json.load(f)
    for event in events_info:
        event_name = event['event_name']
        event_start_time = event['event_start_time']
        event_end_time = event['event_end_time']
        log.info(f'Processing event features: {event_name}')
        event_start_time = datetime.datetime.strptime(event_start_time, '%Y-%m-%d %H:%M')
        rel_filepath = get_rel_filepath(source=data_source, event_start_date=event_start_time)
        org_filepath = get_org_filepath(source=data_source, event_start_date=event_start_time, use_local=True)
        pfx2as_filepath = get_pfx2as_filepath(source="CAIDA", event_start_date=event_start_time, use_local=True)
        asTopology = ASFeature(rel_filename=rel_filepath, org_filename=org_filepath, pfx2as_filename=pfx2as_filepath)


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

def calc_whole_alert_count():
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



if __name__ == '__main__':
    # get_pfx2as_filepath(source="RIPE", event_start_date=datetime.datetime(year=2021, month=6, day=1))
    # get_pfx2as_filepath(source="CAIDA", event_start_date=datetime.datetime(year=2022, month=6, day=1))
    # get_pfx2as_filepath(source="CAIDA", event_start_date=datetime.datetime(year=2023, month=10, day=1))
    # preCalcASFeature(data_source="CAIDA")
    # preCalcASFeature_mp(data_source="CAIDA")
    eval_mydata_optimized(data_source='CAIDA')
    calc_point_wise_metric()