from collections import defaultdict
import datetime
import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd
from tqdm import tqdm
import logging
from zoneinfo import ZoneInfo # py3.9后是系统库
def beijing(sec, what):
    beijing_time = datetime.datetime.now(ZoneInfo('Asia/Shanghai')) # 返回北京时间
    return beijing_time.timetuple()
logging.Formatter.converter = beijing
logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S', format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)
from commons import read_event_data

events_root = '/data/data/anomaly-event-routedata'

def view_anomaly_message():
    with open('events.json', 'r') as f:
        events_info = json.load(f)
    for event in events_info:
        pred_y = []
        event_name = event['event_name']
        event_start_time = event['event_start_time']
        event_end_time = event['event_end_time']

        save_dir = os.path.join('mydata_test_result', event_name)
        # with open(os.path.join(save_dir, 'true_labels.txt'), 'r') as f:
        #     true_labels = eval(f.read())
        with open(os.path.join(save_dir, 'pred_labels.txt'), 'r') as f:
            pred_y = eval(f.read())

        all_msgs, true_y = read_event_data(events_root, event_name, event_start_time, event_end_time)
        with open(os.path.join(save_dir, 'anomaly_messages.txt'), 'w') as f:
            for pred_label, msg, true_label in zip(pred_y, all_msgs, true_y):
                if pred_label:
                    f.write(f'pred:{pred_label} actual:{true_label}|{msg}\n')


# def count_frequency_per_min():
#     anomaly_messages = []
#     with open('mydata_test_result/leak-20041224-TTNet_in_Turkey_leak/anomaly_messages.txt', 'r') as f:
#         anomaly_messages = f.read()
#     anomaly_messages = anomaly_messages.strip().split('\n')
#     start_time = datetime.datetime.strptime(anomaly_messages[0].split('|')[3], "%Y-%m-%d %H:%M:%S")
#     end_time = datetime.datetime.strptime(anomaly_messages[-1].split('|')[3], "%Y-%m-%d %H:%M:%S")
#     time_bins = pd.date_range(start=start_time, end=end_time, freq='1min')
#     # 将事件时间戳转换为numpy数组
#     event_times = np.array([datetime.datetime.strptime(msg.split('|')[3], "%Y-%m-%d %H:%M:%S").timestamp() for msg in anomaly_messages])

#     # 使用numpy的histogram函数统计每个时间段内的事件数量
#     hist, bin_edges = np.histogram(event_times, bins=time_bins)


#     # 绘制直方图
#     plt.hist(event_times, bins=time_bins, edgecolor='black')
#     plt.xlabel('Time')
#     plt.ylabel('Number of Events')
#     plt.title('Event Frequency Histogram')
#     # plt.xticks(rotation=45)
#     # plt.tight_layout()
#     plt.savefig('hist.png')
#     plt.show()

def count_frequency_per_min():
    # 统计消息频率
    anomaly_messages = []

    try:
        # 读取文件内容
        with open('mydata_test_result/leak-20041224-TTNet_in_Turkey_leak/anomaly_messages.txt', 'r') as f:
            anomaly_messages = f.read().strip().split('\n')

        if not anomaly_messages:
            raise ValueError("文件为空或未包含任何数据")

        # 提取事件时间和转换为 datetime 对象
        event_times = []
        for msg in anomaly_messages:
            try:
                # 假设时间字段在每行的第4列（索引3）
                timestamp_str = msg.split('|')[3].strip()
                event_time = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                event_times.append(event_time)
            except (IndexError, ValueError) as e:
                print(f"跳过无效行: {msg}，错误: {e}")
                continue

        if not event_times:
            raise ValueError("未提取到有效的时间数据")

        # 定义时间范围和分箱
        start_time = min(event_times)
        end_time = max(event_times)
        time_bins = pd.date_range(start=start_time, end=end_time, freq='1min')

        # 将事件时间戳转换为时间戳（秒级）
        event_times_timestamps = np.array([t.timestamp() for t in event_times])
        time_bins_timestamps = np.array([t.timestamp() for t in time_bins])

        # 使用 numpy 的 histogram 函数统计每个时间段内的事件数量
        hist, bin_edges = np.histogram(event_times_timestamps, bins=time_bins_timestamps)

        # 绘制直方图
        plt.figure(figsize=(16, 6))
        plt.bar(
            [datetime.datetime.fromtimestamp(t) for t in bin_edges[:-1]],  # 转换回 datetime 格式
            hist,
            width=1 / 720,  # 每分钟宽度
            align='edge',
            color='skyblue',
            edgecolor='black'
        )

        # 设置横轴为时间格式
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MinuteLocator(interval=5))  # 每5分钟一个刻度

        # 添加标题和标签
        plt.title('Event Frequency Histogram (Per Minute)', fontsize=14)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Number of Events', fontsize=12)
        plt.xticks(rotation=45, fontsize=4)  # 旋转横轴标签以避免重叠
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # 保存图像
        plt.tight_layout()
        plt.savefig('hist.png', dpi=500)  # 设置分辨率为300 DPI
        plt.show()

    except FileNotFoundError:
        print("文件未找到，请检查路径是否正确")
    except Exception as e:
        print(f"发生错误: {e}")


def calc_alert_proportion():
    # 计算告警消息在时间段（1min）内的占比
    with open('events.json', 'r') as f:
        events_info = json.load(f)
    for event in events_info:
        event_name = event['event_name']
        event_start_time = event['event_start_time']
        event_end_time = event['event_end_time']
        save_dir = os.path.join('mydata_test_result', event_name)
        with open(os.path.join(save_dir, 'pred_labels.txt'), 'r') as f:
            pred_y = eval(f.read())
        # with open(os.path.join(save_dir, 'true_labels.txt'), 'r') as f:
        #     true_y = eval(f.read())
        all_msgs, true_y = read_event_data(events_root, event_name, event_start_time, event_end_time)
        last_time = None
        last_idx = 0
        alert_proportion = []
        x_datetimes = []
        for idx, msg in enumerate(tqdm(all_msgs)):
            msg = msg.split('|')
            event_time = datetime.datetime.strptime(msg[1], "%Y-%m-%d %H:%M:%S")
            if last_time is None:
                last_time = event_time
            if event_time - last_time > datetime.timedelta(minutes=5) or idx==len(all_msgs)-1:
                alert_count = sum(pred_y[last_idx:idx+1])
                total_count = idx+1-last_idx
                proportion = alert_count / total_count
                alert_proportion.append(proportion)
                x_datetimes.append(last_time)
                # print(f'Event: {event_name}, Alert Proportion: {proportion}')
                
                last_idx = idx
                last_time = event_time
        # 绘制直方图
        plt.figure(figsize=(16, 6))  # 设置图形大小
        plt.bar(
            x_datetimes,
            alert_proportion,
            width=1 / 720,  # 每分钟宽度
            align='edge',
            color='skyblue',
            edgecolor='black'
        )

        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MinuteLocator(interval=5))  # 每5分钟一个刻度
        # 添加标题和标签
        plt.title('Histogram of Alert Proportion', fontsize=14)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Proportion', fontsize=12)
        plt.xticks(rotation=45, fontsize=4)  # 旋转横轴标签以避免重叠

        # 显示网格
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # 保存图像（可选）
        plt.tight_layout()
        os.makedirs(os.path.join(save_dir, "charts"), exist_ok=True)
        plt.savefig(os.path.join(save_dir, "charts", 'Alert Proportion.png'), dpi=500)  # 设置分辨率为300 DPI
        

def merge_triplets_graph(triplets):
    if len(triplets)==1:
        return triplets
    # 创建有向图
    G = nx.DiGraph()
    for triplet in triplets:
        a, b, c = triplet
        G.add_edges_from([(a, b), (b, c)])
    
    # 找到所有弱连通分量（忽略边的方向）
    components = list(nx.weakly_connected_components(G))
    merged = set()
    
    for component in components:
        subgraph = G.subgraph(component)
        
        # 直接进行拓扑排序（假设子图是DAG）
        try:
            ordered_nodes = list(nx.topological_sort(subgraph))
        except nx.NetworkXUnfeasible:
            # 如果存在环，返回空列表（根据用户要求不处理环）
            ordered_nodes = []
        merged.add(tuple(ordered_nodes))
        # 去重并保持拓扑顺序
        # seen = set()
        # ordered_unique = []
        # for node in ordered_nodes:
        #     if node not in seen:
        #         seen.add(node)
        #         ordered_unique.append(node)
        # if ordered_unique:
        #     merged.add(tuple(ordered_unique))
    return merged

def merge_anomaly_msgs(anomaly_msgs):
    """告警压缩

    Args:
        anomaly_msgs (tuple): [[pred_label, true_label, triplets, msg], ...]

    Returns:
        dict: (leaker, leaked_to):[[aspath, timestamp],...]
    """
    # 合并三元组和as path一样的消息
    result = []
    prev_triplets = None
    prev_path = None
    log.info('开始合并相同告警')
    # todo:后面计数时候是否要考虑这里合并掉了许多消息？
    for pred_label, true_label, triplets, msg in tqdm(anomaly_msgs):
        # 如果 triplets 不同，则添加到结果中
        if triplets != prev_triplets and prev_path != msg.split('|')[6]:
            result.append((pred_label, true_label, triplets, msg))
            prev_triplets = triplets
            prev_path = msg.split('|')[6]
    anomaly_msgs = result
    # 合并多个三元组为一个
    result_dict = []
    tuple_dict = {} # 最终结果二元组，对应所有的aspath和时间戳
    log.info('开始提取告警消息')
    for pred_label, true_label, triplets, msg in tqdm(anomaly_msgs):
        merged_triplets = merge_triplets_graph(triplets)
        tmp = set()
        for t in merged_triplets:
            if not t:
                continue
            tmp.add(t)
        merged_triplets = tmp
        if not merged_triplets:
            continue
        ASPath = msg.split('|')[6] 
        # EventStartTimestamp = datetime.datetime.strptime(msg.split('|')[1], "%Y-%m-%d %H:%M:%S").timestamp()
        EventStartTimestamp = int(msg.split('|')[1])
        # 取后三个as作为泄漏者
        if len(merged_triplets)>1:
            # 寻找aspath中最靠后的元组
            as_path = ASPath.split()
            # 查找完全匹配的元组, 如果无法完全匹配(as path被清洗过)，则匹配开头as
            last_index = -1
            last_matching_tuple = None
            last_index_1 = -1 # 匹配的开头as idx
            last_matching_tuple_1 = None
            for t in merged_triplets:
                t_len = len(t)
                for i in range(len(as_path) - t_len + 1):
                    if as_path[i:i + t_len] == list(t):
                        if i + t_len - 1 > last_index:
                            last_index = i + t_len - 1
                            last_matching_tuple = t
                    elif as_path[i] == t[0] and i + t_len - 1 > last_index_1:
                        last_index_1 = i + t_len - 1
                        last_matching_tuple_1 = t
            target_tuple = None
            if last_index>last_index_1 and last_matching_tuple:
                LeakerAS = last_matching_tuple[-2]
                LeakedTo = last_matching_tuple[-3]
                target_tuple = last_matching_tuple[-3:]
            elif last_index<last_index_1 and last_matching_tuple_1:
                LeakerAS = last_matching_tuple_1[-2]
                LeakedTo = last_matching_tuple_1[-3]
                target_tuple = last_matching_tuple_1[-3:]
            else:
                log.error(f"查找Leaker失败, triplets:{triplets}, merged_triplets:{merged_triplets}, ASPath:{ASPath}, ")
                continue
        else:
            triplet = list(merged_triplets)[0]
            LeakerAS = triplet[-2]
            LeakedTo = triplet[-3]
            target_tuple = triplet[-3:]
        # result_dict.append({
        #     'LeakerAS': LeakerAS,
        #     'LeakedTo': LeakedTo,
        #     'ASPath': ASPath,
        #     'EventStartTimestamp': EventStartTimestamp
        # })
        tuple_dict.setdefault((LeakedTo, LeakerAS), []).append([ASPath, EventStartTimestamp])
        # tuple_dict.setdefault(target_tuple, []).append([ASPath, EventStartTimestamp])

    tuple_dict = dict(sorted(tuple_dict.items(), key=lambda x:(len(x[1]), x[0]), reverse=True))
    return tuple_dict


def view_alert():
    with open('events.json', 'r') as f:
        events_info = json.load(f)
    for event in events_info:
        pred_y = []
        event_name = event['event_name']
        event_start_time = event['event_start_time']
        event_end_time = event['event_end_time']
        log.info(f'Processing event: {event_name}')
        save_dir = os.path.join('mydata_test_result', event_name)
        # with open(os.path.join(save_dir, 'true_labels.txt'), 'r') as f:
        #     true_labels = eval(f.read())
        with open(os.path.join(save_dir, 'pred_labels.txt'), 'r') as f:
            pred_y = eval(f.read())

        all_msgs, true_y = read_event_data(events_root, event_name, event_start_time, event_end_time)
        anomaly_msgs=[]
        for pred_label, msg, true_label in tqdm(zip(pred_y, all_msgs, true_y)):
            if pred_label:
                anomaly_msgs.append([pred_label, true_label, triplets, msg])
        # 告警压缩
        anomaly_tuples = merge_anomaly_msgs(anomaly_msgs) # 异常元组(LeakedTo, LeakerAS)
        for k, v in list(anomaly_tuples.items())[:10]:
            print(f'leaker:{k[0]}, leaked_to:{k[1]}, as_path_num:{len(v)}')


if __name__ == '__main__':
    view_alert()
    # calc_alert_proportion()
    with open('mydata_test_result/leak-20041224-TTNet_in_Turkey_leak/anomaly_messages.txt', 'r') as f:
        tmp = f.read().strip().split('\n')
    anomaly_messages = []
    
    for msg in tqdm(tmp):
        msg = msg.split('|')
        pred_label = int(msg[0][5])
        true_label = int(msg[0][14])
        triplets = set()
        matches = re.findall(r"\((\d+),(\d+),(\d+)\)", msg[1])
        for x in matches:
            triplets.add(x)
        anomaly_messages.append([pred_label, true_label, triplets, '|'.join(msg[2:])])
    anomaly_messages = merge_anomaly_msgs(anomaly_messages)

