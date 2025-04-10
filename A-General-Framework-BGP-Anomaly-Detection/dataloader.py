from collections import defaultdict
import datetime
import json
import os
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

from commons import read_event_list
def dd():
    return defaultdict(int)  
def ds():
    return defaultdict(set)
def dl():
    return defaultdict(list)

class SingleEventDataset(Dataset):
    def __init__(self, event_path, event_start_time, event_end_time, victim_AS, force_reCalc=False):
        """针对单个时间的Dataset

        Args:
            event_path (str): 事件路径
            event_start_time (str): "%Y-%m-%d %H:%M"
            event_end_time (str): "%Y-%m-%d %H:%M"
            victim_AS (int): 实际上是泄露者as
            force_reCalc (bool): 重新计算features
        """
        self.event_path = event_path
        self.event_start_time = datetime.datetime.strptime(event_start_time, "%Y-%m-%d %H:%M").timestamp()
        if event_end_time != "":
            self.event_end_time = datetime.datetime.strptime(event_end_time, "%Y-%m-%d %H:%M").timestamp()
        else:
            self.event_end_time = 9999999999
        self.victim_AS = victim_AS
        self.features = pd.DataFrame()
        self.labels = pd.DataFrame()
        self.raw_data:list[str] = []
        event_name = os.path.split(self.event_path)[-1]
        feature_filepath = os.path.join('mydata_features', f'{event_name}.csv')
        if os.path.exists(feature_filepath) and not force_reCalc:
            data = pd.read_csv(feature_filepath, index_col=0)
            # self.data_cols = pd.read_csv('./result_doc/data_all.csv', index_col=0).columns
            # # 为缺失的列赋默认值（例如 NaN）
            # for col in set(self.data_cols)-set(data.columns):
            #     # if col.startswith('label_'):
            #     #     continue
            #     data[col] = 0  # 可以替换为其他默认值，如 0
            self.labels = data['label']
            # data = data.drop(columns=['label'], axis=1)
            # data.reindex(columns=self.data_cols)
            # self.features = data.drop(columns=['label_0', 'label_1', 'label_2', 'label_3'], axis=1)
            self.features = data.drop(columns=['label'], axis=1)
        else:
            for root, dirs, files in os.walk(os.path.join(self.event_path, 'decoded')):
                dirs.sort()
                files.sort()
                for filename in files:
                    filepath = os.path.join(root, filename)
                    with open(filepath, 'r') as f:
                        tmp = f.read().strip().split('\n')
                        self.raw_data.extend(tmp)
            
            self.features, self.labels = FeatureExtractor(self.raw_data, self.event_start_time, self.event_end_time, self.victim_AS).ExtractFeatures()

            data = pd.concat([self.features, self.labels], axis=1)
            data.to_csv(feature_filepath)

        # 正负样本均衡
        # features = self.features.values.tolist()
        # labels = self.labels.values.tolist()
        # print('Balancing Samples')
        # rate = 1 # 负样本/正样本
        # pos_num = labels.count(1)
        # pos_indices = np.where(np.array(labels) == 1)[0].tolist()
        # neg_indices = np.where(np.array(labels) == 0)[0].tolist()
        # np.random.seed(42)  # 固定随机种子以确保结果可复现
        # sampled_neg_indices = np.random.choice(neg_indices, size=min(len(neg_indices), int(rate * pos_num)), replace=False)
        # # 合并正负样本下标
        # selected_indices = pos_indices + sampled_neg_indices.tolist()
        # selected_indices.sort()
        # selected_feature = []
        # for i in tqdm(selected_indices):
        #     selected_feature.append(features[i])
        # selected_labels = np.array(labels)[selected_indices].tolist()
        # features = selected_feature
        # labels = selected_labels
        # self.features = pd.DataFrame(features, columns=self.features.columns)
        # self.labels = pd.DataFrame(labels, columns=['label'])
            
        self.scaler:MinMaxScaler = pickle.load(open('./params/route_leak_scaler.pkl', 'rb'))
        new_features = self.scaler.fit_transform(np.array(self.features))
        self.features = pd.DataFrame(new_features, columns=self.features.columns, index=self.features.index)
        self.features, self.labels = FeatureExtractor.to_timestep(self.features, self.labels)

    
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # feature = self.features.iloc[idx]
        # label = self.labels.iloc[idx]
        feature = self.features[idx]
        label = self.labels[idx]
        # feature = self.scaler.fit_transform(np.array(feature))
        feature = torch.tensor(feature, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.int)
        return feature, label
    
class FeatureExtractor:
    def __init__(self, raw_data, event_start_time, event_end_time, victim_AS):
        self.raw_data:list[str] = raw_data
        self.event_start_time = event_start_time
        self.event_end_time = event_end_time
        self.victim_AS = victim_AS
        self.features = pd.DataFrame()
        self.labels = pd.DataFrame()


    def ExtractFeatures(self, ):
        self.SplitTimeline()
        self.post_process()
        return self.features, self.labels

    def SplitTimeline(self, ):
        """Step1: Feature_Extractor.py中的get_his_info部分
        主要作用为按1分钟为间隔切割时间轴, 计算特征
        """
        Window_size = 2  # min

        # for i in range(total_num):
        event_happening = False
        VICTIM_AS = self.victim_AS
        step = 0

        # <prefix, origin-ASns-set > dictionary
        path = set()  
        victim_prefix = set()  
        MOAS = defaultdict(set)  
        old_time = 0
        first = True
        peer = set()  # victime's peer
        peer_num = 0  

        peer_increase = defaultdict(int)  
        features = pd.DataFrame()  # [timestamp][feature][values]

        # prefix_origin = defaultdict(set)   #prefix origin pairs
        temp = list

        # feature
        MPL = defaultdict(int)  # MAX path length in time t
        PL = defaultdict(dd)  # each number of path len in time t.
        MOAS_AS = defaultdict(ds)  # Number of ASes that conflict with victime AS
        old_AS_Path = defaultdict(list)  
        new_AS_Path = defaultdict(list)  
        diff_AS_Path = defaultdict(dd)  
        diff_AS_Path_num = defaultdict(dd)  # AS-PATH edit distance set
        withdraw_num = defaultdict(int)
        new_sub_prefix_num = defaultdict(int)  # Number of new sub-prefixes belongs to Victim AS
        new_sub_prefix = defaultdict(set)  
        own_ann_num = defaultdict(int)  # Number of Announcements from victim AS
        MOAS_ann_num = defaultdict(int)  # Number of announcements from origin conflict AS
        ann_shorter = defaultdict(dd)  # AS-PATH length decrease set
        ann_longer = defaultdict(dd)  # AS-PATH length increase set

        Diff_Ann = defaultdict(int)
        duplicate_ann = defaultdict(int)
        withdraw_unique_prefix = defaultdict(set)
        # IGP_num=defaultdict(int)
        # EGP_num=defaultdict(int)
        # incomplete_packets=defaultdict(int)
        new_MOAS = defaultdict(int)

        avg_edit_distance = 0

        avg_PL = 0

        # 标签
        labels = defaultdict(dd)
        for msg_idx, msg in enumerate(tqdm(self.raw_data)):
            # if msg_idx/len(self.raw_data)*100 % 5 == 0:
            #     print(f'进度：{msg_idx}/{len(self.raw_data)}')
            msg = msg.split('|')
            # msg[1] = int(datetime.datetime.strptime(msg[1], "%Y-%m-%d %H:%M:%S").timestamp())
            msg[1] = int(msg[1])

            # Get the prefix
            if (first == True):
                old_time = msg[1]
                first = False
            pfx = msg[5]
            # Get the list of ASes in the AS path
            # print(elem)
            if (msg[2] == 'A'):
                ases = msg[6].split(" ")

                len_path = len(ases)  # AS-PATH len

                if len_path > 0:
                    # Get the origin ASn (rightmost)
                    # modify: 考虑有花括号的情况
                    origin = ases[-1]
                    if origin.startswith('{'):
                        origins = list(map(int, origin[1:-1].split(',')))
                    else:
                        origins = [int(origin)]
                    if (VICTIM_AS in origins):
                        own_ann_num[old_time] += 1
                        if pfx not in victim_prefix:
                            for father_pfx in victim_prefix:  # if it's the new_subprefix
                                if self.is_sub_pfx(father_pfx, pfx):
                                    new_sub_prefix_num[old_time] += 1
                                    new_sub_prefix[old_time].add(pfx)
                                    break
                            victim_prefix.add(pfx)
                        peer = ases[0]

                        if peer not in new_AS_Path.keys():
                            peer_num += 1
                            peer_increase[old_time] += 1
                            new_AS_Path[peer] = ases
                            path_str = 'len_path' + str(len_path)
                            PL[old_time][path_str] += 1
                            if (len_path > MPL[old_time]):
                                MPL[old_time] = len_path
                        else:
                            if ases != new_AS_Path[peer]:  # if path change, calculate it's edit distance
                                Diff_Ann[old_time] += 1
                                old_AS_Path[peer] = new_AS_Path[peer]
                                new_AS_Path[peer] = ases
                                num, len_cut = self.edit_distance(old_AS_Path[peer], new_AS_Path[peer])
                                if (len_cut > 0):
                                    ann_shorter_str = 'ann_shorter_' + str(len_cut)
                                    ann_shorter[old_time][ann_shorter_str] += 1
                                else:
                                    ann_longer_str = 'ann_longer_' + str(-len_cut)
                                    ann_longer[old_time][ann_longer_str] += 1
                                diff_num = 'diff_' + str(num)
                                # diff_peer = 'diff_peer_' + str(peer)
                                diff_AS_Path_num[old_time][diff_num] += 1
                                # diff_AS_Path[old_time][diff_peer] = num
                                path_str = 'len_path' + str(len_path)
                                PL[old_time][path_str] += 1
                                if (len_path > MPL[msg[1]]):
                                    MPL[old_time] = len_path

                            else:
                                duplicate_ann[old_time] += 1
                        # print(elem.fields["as-path"])
                        # print(pfx)
                    # Insert the origin ASn in the set of
                    # origins for the prefix
                    else:
                        if pfx in victim_prefix:
                            MOAS_ann_num[old_time] += 1
                            if origin not in MOAS:
                                new_MOAS[old_time] += 1
                                MOAS[old_time].add(origin)

                            MOAS_AS[old_time][pfx].add(origin)

            elif (msg[2] == 'W'):
                if pfx in victim_prefix:
                    withdraw_num[old_time] += 1
                    withdraw_unique_prefix[old_time].add(pfx)

            if (msg[1] >= (old_time + 30 * Window_size)):
                label = 0
                if (abs(old_time - self.event_start_time) < 30) and not event_happening:  # label our date
                    event_happening = True
                elif event_happening and old_time - self.event_end_time > 30:
                    event_happening = False
                # print(abs(old_time-start_time))
                if event_happening:
                    label = 1
                else:
                    label = 0
                labels[old_time]['label'] = label
                df = pd.DataFrame({'time': pd.to_datetime(old_time, unit='s'),
                                    'MPL': MPL[old_time],
                                    'MOAS_prefix_num': len(MOAS_AS[old_time]),
                                    'MOAS_AS': [MOAS_AS[old_time]],
                                    'MOAS': [MOAS[old_time]],
                                    'new_MOAS': new_MOAS[old_time],
                                    'MOAS_num': len(MOAS[old_time]),
                                    'withdraw_num': withdraw_num[old_time],
                                    'peer_increase': peer_increase[old_time],
                                    'peer_num': peer_num,
                                    'new_prefix_num': new_sub_prefix_num[old_time],
                                    'MOAS_Ann_num': MOAS_ann_num[old_time],
                                    'own_Ann_num': own_ann_num[old_time],
                                    'new_sub_prefix': [new_sub_prefix[old_time]],
                                    'Victim_AS': VICTIM_AS,
                                    'Diff_Ann': Diff_Ann[old_time],
                                    'duplicate_ann': duplicate_ann[old_time],
                                    'withdraw_unique_prefix_num': len(withdraw_unique_prefix[old_time]),
                                    'withdraw_unique_prefix': [withdraw_unique_prefix[old_time]],
                                    }, index=[old_time])
                d1 = pd.DataFrame(diff_AS_Path_num[old_time], index=[old_time])
                # d2=pd.DataFrame(diff_AS_Path[old_time],index=[old_time])
                d2 = pd.DataFrame(labels[old_time], index=[old_time])
                d3 = pd.DataFrame(PL[old_time], index=[old_time])
                d5 = pd.DataFrame(ann_shorter[old_time], index=[old_time])
                d6 = pd.DataFrame(ann_longer[old_time], index=[old_time])

                # df2=pd.concat([d1,d3],axis=1)
                d4 = pd.concat([df, d1, d2, d3, d5, d6], axis=1)
                # print(d4)

                features = pd.concat([features, d4])
                old_time = msg[1]
        # print(features)
        # print(features['label_0'])
        self.features = features
        # features.to_csv(os.path.join(self.event_path, 'IspSelfOpFeature.csv'))
    
    def post_process(self,):
        """Data_Loader的load函数
        """
        # data prepocessing
        data_all = self.features.drop(
            columns=['time', 'new_sub_prefix', 'MOAS_AS', 'Victim_AS', 'MOAS', 'withdraw_unique_prefix'],
            axis=1)

        data_all.fillna(0, inplace=True)

        self.__add_count(data_all, 14, 21, 11, 11)
        self.data_cols = pd.read_csv('./result_doc/data_all.csv', index_col=0).columns
        # 为缺失的列赋默认值（例如 NaN）
        for col in set(self.data_cols)-set(data_all.columns):
            data_all[col] = 0  # 可以替换为其他默认值，如 0
        data_all.fillna(0, inplace=True)

        self.labels = data_all['label']
        data_all = data_all.drop(columns=['label'], axis=1)
        data_all.fillna(0, inplace=True)
        data_all.reindex(columns=self.data_cols)
        self.features = data_all.drop(columns=['label_0', 'label_1', 'label_2', 'label_3'], axis=1)
        # print(data_all)

        # change test features to train features
        # x, y0, y1, y2, y3 = data_all.drop(columns=['label_0', 'label_1', 'label_2', 'label_3'], axis=1), \
        #                     data_all['label_0'], data_all['label_1'], data_all['label_2'], data_all['label_3']
        # return x, y0
        # self.features = data_all.drop(columns=['label'], axis=1)

    @staticmethod
    def to_timestep(x_raw, y_raw, ):
        WINDOW_SIZE = 30
        TIME_STEP = 1
        event_len = len(y_raw)
        # X=np.array()
        x=np.array(x_raw)
        y=np.array(y_raw)
        Y:list = []  # type: List[Any]

        for step in range(0, event_len - WINDOW_SIZE + 1, TIME_STEP):
            if step == 0:
                tempx = x[step:step + WINDOW_SIZE, :]
                tempx = tempx[np.newaxis, :]
                X = tempx
                Y.append(y[step + WINDOW_SIZE - 1])
            else:
                tempx = x[step:step + WINDOW_SIZE, :]
                tempx = tempx[np.newaxis, :]
                X = np.concatenate((X, tempx), axis=0)
                Y.append(y[step + WINDOW_SIZE - 1])
        # print(len(X))
        print(X.shape)

        # print(x.head)
        return X, Y  # return array

    def edit_distance(self,l1, l2):  # 
        """

        :param l1: list 1
        :param l2: list 2
        :return: edit distance between l1 and l2
        """
        rows = len(l1) + 1
        cols = len(l2) + 1

        dist = [[0 for x in range(cols)] for x in range(rows)]

        for i in range(1, rows):
            dist[i][0] = i
        for i in range(1, cols):
            dist[0][i] = i

        for col in range(1, cols):
            for row in range(1, rows):
                if l1[row - 1] == l2[col - 1]:
                    cost = 0
                else:
                    cost = 1
                dist[row][col] = min(dist[row - 1][col] + 1,  # deletion
                                     dist[row][col - 1] + 1,  # insertion
                                     dist[row - 1][col - 1] + cost)  # substitution
        return dist[row][col], rows - cols

    def is_sub_pfx(self,father_prefix, sub_prefix):
        father_pfx, f_mask = father_prefix.split('/')
        sub_pfx, sub_mask = sub_prefix.split('/')
        f_mask = int(f_mask)
        sub_mask = int(sub_mask)
        if (f_mask > sub_mask):
            return False
        elif (f_mask % 8 == 0):  # 
            block = int(f_mask / 8)
            list1 = father_pfx.split('.')
            list2 = sub_pfx.split('.')
            return (list1[0:block] == list2[0:block])
        else:  # 
            father_IP_bin = self.to_bin(father_pfx)
            sub_IP_bin = self.to_bin(sub_pfx)
            father_IP_bin = father_IP_bin[0:f_mask]
            sub_IP_bin = sub_IP_bin[0:f_mask]
            if father_IP_bin == sub_IP_bin:
                return True
            else:
                return False
    
    def to_bin(self,IP):
        list1 = IP.split('.')
        list2 = []
        for item in list1:
            try:
                item = bin(int(item))  # ---0b11000000 0b10101000 0b1100000 0b1011110 ----

                # cut the first 2 bin :0b.
                item = item[2:]
            except:
                print(IP)

            list2.append(item.zfill(8))  # --['11000000', '10101000', '01100000', '01011110']--
        v2 = ''.join(list2)  # ----11000000101010000110000001011110----
        # print(v2)
        return v2

    def __add_count(self,data_all:pd.DataFrame, edit_threshold, pl_threshold, longer_threshold, shorter_threshold):
        import re

        ls = data_all.columns

        diff_group = []
        len_group = []
        ann_longer_group = []
        ann_shorter_group = []
        drop_edit = set()
        drop_pl = set()
        drop_longer = set()
        drop_shorter = set()
        ls = data_all.columns
        for col in ls:
            if re.search(r"diff_\d+", col) != None:
                # print(re.search("diff_\d+", col).string)
                diff_group.append(re.search(r"diff_\d+", col).string)
            elif re.search(r"len_path\d+", col) != None:
                # print(re.search("len_path\d+", col).string)
                len_group.append(re.search(r"len_path\d+", col).string)
            elif re.search(r"ann_longer_\d+", col) != None:
                # print(re.search("ann_longer_\d+", col).string)
                ann_longer_group.append(re.search(r"ann_longer_\d+", col).string)
            elif re.search(r"ann_shorter_\d+", col) != None:
                ann_shorter_group.append(re.search(r"ann_shorter_\d+", col).string)
        data_all['sum_diff'] = 0
        data_all['sum_diff_num'] = 0
        data_all['PL_sum'] = 0
        data_all['sum_len_num'] = 0
        data_all['sum_ann_longer'] = 0
        data_all['sum_ann_longer_num'] = 0
        data_all['sum_ann_shorter'] = 0
        data_all['sum_ann_shorter_num'] = 0
        data_all['avg_diff'] = 0
        data_all['avg_longer'] = 0
        data_all['avg_shorter'] = 0
        data_all['avg_len'] = 0
        edit = 'edit_bigger_' + str(edit_threshold)
        ppl = 'PL_bigger_' + str(pl_threshold)
        longer = 'longer_bigger_' + str(longer_threshold)
        shorter = 'shorter_bigger_' + str(shorter_threshold)
        data_all[edit] = 0
        data_all[ppl] = 0
        data_all[longer] = 0
        data_all[shorter] = 0
        for diff in diff_group:
            num = int(diff.split('_')[1])
            data_all['sum_diff'] += num * data_all[diff]
            data_all['sum_diff_num'] += data_all[diff]
            if num >= edit_threshold:
                data_all[edit] += data_all[diff]
                drop_edit.add(diff)
        for PL in len_group:
            num = int(PL.split('h')[1])
            data_all['PL_sum'] += num * data_all[PL]
            data_all['sum_len_num'] += data_all[PL]
            if num >= pl_threshold:
                drop_pl.add(PL)
                data_all[ppl] += data_all[PL]
        for al in ann_longer_group:
            num = int(al.split('_')[2])
            data_all['sum_ann_longer'] += num * data_all[al]
            data_all['sum_ann_longer_num'] += data_all[al]
            if num >= longer_threshold:
                data_all[longer] += data_all[al]
                drop_longer.add(al)
        for ann_shorter in ann_shorter_group:
            num = int(ann_shorter.split('_')[2])
            data_all['sum_ann_shorter'] += num * data_all[ann_shorter]
            data_all['sum_ann_shorter_num'] += data_all[ann_shorter]

            if num >= shorter_threshold:
                data_all[shorter] += data_all[ann_shorter]
                drop_shorter.add(ann_shorter)

            data_all['avg_diff'] = data_all['sum_diff'] / data_all['sum_diff_num']

            data_all['avg_longer'] = data_all['sum_ann_longer'] / data_all['sum_ann_longer_num']

            data_all['avg_shorter'] = data_all['sum_ann_shorter'] / data_all['sum_ann_shorter_num']

            data_all['avg_len'] = data_all['PL_sum'] / data_all['sum_len_num']
        data_all.drop(columns=list(drop_edit), inplace=True)
        data_all.drop(columns=list(drop_pl), inplace=True)
        data_all.drop(columns=list(drop_longer), inplace=True)
        data_all.drop(columns=list(drop_shorter), inplace=True)

    
def fill_resort_columns(event_name, target_columns):
    feature_filepath = os.path.join('mydata_features_o', f'{event_name}.csv')
    data = pd.read_csv(feature_filepath, index_col=0)
    labels = data['label']
    # 为缺失的列赋默认值（例如 NaN）
    for col in set(target_columns)-set(data.columns):
        data[col] = 0  # 可以替换为其他默认值，如 0
    data.drop(columns=['label'], axis=1, inplace=True)
    data.fillna(0, inplace=True)
    data.reindex(columns=target_columns)
    data.drop(columns=['label_0', 'label_1', 'label_2', 'label_3'], axis=1, inplace=True)
    data = pd.concat([data, labels], axis=1)
    data.to_csv(os.path.join('mydata_features', f'{event_name}.csv'))


if __name__ == '__main__':
    # with open('events.json', 'r') as f:
    #     events = json.load(f)
    #     data_cols = pd.read_csv('./result_doc/data_all.csv', index_col=0).columns
    # events = read_event_list(evnet_type='hijack')
    events = read_event_list(evnet_type='leak')
    for event in events:
        event_name = event['event_name']
        event_start_time = event['event_start_time']
        event_end_time = event['event_end_time']
        leaker_as = event['leaker_as']
        print(f'processing event:{event_name}')
        # fill_resort_columns(event_name, data_cols)
        dataset = SingleEventDataset(
            event_path=f"/data/data/anomaly-event-routedata/{event_name}", 
            event_start_time=event_start_time,
            event_end_time=event_end_time,
            victim_AS=leaker_as,
            # force_reCalc=True
        )
        # dataset.__len__()
        # t=dataset.__getitem__(360)
        # print(t[0])