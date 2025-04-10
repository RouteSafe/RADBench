import numpy as np
from collections import defaultdict, Counter
import os
from datetime import datetime, timedelta
import pytz
from pathlib import Path
import pandas as pd
from multiprocessing import Pool, cpu_count
import subprocess
from ipaddress import IPv4Network, IPv6Network
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import threading
import gc
import logging
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bgp_feature_extractor.log'),
        logging.StreamHandler()
    ]
)

# 添加线程锁，用于日志输出
log_lock = threading.Lock()

def log_info(message):
    """线程安全的日志记录"""
    with log_lock:
        logging.info(message)

class ASPathAnalyzer:
    def __init__(self):
        # 使用基本数据结构
        self.prefix_to_as = defaultdict(set)
        self.prefix_as_paths = defaultdict(list)
        self.moas_prefixes = set()
        self.moas_ases = set()
        self.known_prefixes = set()
        self.new_sub_prefixes = set()
        self.prefix_networks = {}
    
    def get_network(self, prefix):
        """简化的网络对象获取"""
        if prefix not in self.prefix_networks:
            try:
                self.prefix_networks[prefix] = IPv4Network(prefix) if '.' in prefix else IPv6Network(prefix)
            except Exception:
                return None
        return self.prefix_networks[prefix]
        
    def is_sub_prefix(self, prefix, other_prefix):
        """简化的子前缀检查"""
        # 快速预检查
        if prefix == other_prefix:
            return False
            
        try:
            prefix_len = int(prefix.split('/')[1])
            other_len = int(other_prefix.split('/')[1])
            
            if prefix_len <= other_len:
                return False
                
            net1 = self.get_network(prefix)
            net2 = self.get_network(other_prefix)
            if not net1 or not net2:
                return False
                
            return net1.overlaps(net2)
        except:
            return False
            
    def update_as_info(self, prefix, as_path, is_withdrawal=False):
        """简化的AS信息更新"""
        if not is_withdrawal and as_path:
            origin_as = as_path[-1]
            
            # 子前缀检查
            if prefix not in self.known_prefixes:
                try:
                    prefix_len = int(prefix.split('/')[1])
                    for known_prefix in self.known_prefixes:
                        try:
                            known_len = int(known_prefix.split('/')[1])
                            if known_len < prefix_len and self.is_sub_prefix(prefix, known_prefix):
                                self.new_sub_prefixes.add(prefix)
                                break
                        except:
                            continue
                except:
                    pass
                    
                self.known_prefixes.add(prefix)
            
            # MOAS检测
            current_origins = self.prefix_to_as[prefix]
            if origin_as not in current_origins:
                current_origins.add(origin_as)
                if len(current_origins) > 1:
                    self.moas_prefixes.add(prefix)
                    self.moas_ases.update(current_origins)
            
            # 保存路径信息
            self.prefix_as_paths[prefix].append(as_path)
            # 限制历史记录
            if len(self.prefix_as_paths[prefix]) > 20:  # 减少历史记录
                self.prefix_as_paths[prefix] = self.prefix_as_paths[prefix][-20:]

    def calculate_path_changes(self, prefix):
        """简化的路径变化计算"""
        paths = self.prefix_as_paths[prefix]
        if len(paths) < 2:
            result = {
                'shorter_num': 0, 'longer_num': 0,
                'max_shorter': 0, 'max_longer': 0,
                'avg_shorter': 0, 'avg_longer': 0,
                'mid_shorter': 0, 'mid_longer': 0,
                'var_shorter': 0, 'var_longer': 0,
                'mode_shorter': 0, 'mode_longer': 0
            }
            return result
            
        # 计算路径长度变化
        path_lengths = [len(path) for path in paths]
        length_changes = []
        for i in range(1, len(path_lengths)):
            length_changes.append(path_lengths[i] - path_lengths[i-1])
        
        shorter_changes = [-c for c in length_changes if c < 0]
        longer_changes = [c for c in length_changes if c > 0]
        
        # 计算统计值
        result = {
            'shorter_num': len(shorter_changes),
            'longer_num': len(longer_changes),
            'max_shorter': max(shorter_changes) if shorter_changes else 0,
            'max_longer': max(longer_changes) if longer_changes else 0,
            'avg_shorter': sum(shorter_changes)/len(shorter_changes) if shorter_changes else 0,
            'avg_longer': sum(longer_changes)/len(longer_changes) if longer_changes else 0,
            'mid_shorter': sorted(shorter_changes)[len(shorter_changes)//2] if shorter_changes else 0,
            'mid_longer': sorted(longer_changes)[len(longer_changes)//2] if longer_changes else 0,
            'var_shorter': np.var(shorter_changes) if shorter_changes else 0,
            'var_longer': np.var(longer_changes) if longer_changes else 0,
            'mode_shorter': self._simple_mode(shorter_changes),
            'mode_longer': self._simple_mode(longer_changes)
        }
        
        return result
    
    def _simple_mode(self, arr):
        """简化的众数计算"""
        if not arr:
            return 0
        counter = Counter(arr)
        return counter.most_common(1)[0][0]

    def get_features(self):
        """简化的特征提取，增加多线程处理"""
        # 基本特征
        features = {
            'moas_ann_num': len(self.moas_prefixes),
            'moas_as_num': len(self.moas_ases),
            'new_sub_prefix_num': len(self.new_sub_prefixes),
            'moas_prefix_num': len(self.moas_prefixes)
        }
        
        # 路径变化特征
        all_changes = defaultdict(list)
        valid_prefixes = [p for p in self.prefix_as_paths if len(self.prefix_as_paths[p]) >= 2]
        
        # 使用多线程处理大量前缀
        if len(valid_prefixes) > 100:
            # 创建线程池
            with ThreadPoolExecutor(max_workers=8) as executor:
                # 提交任务
                future_to_prefix = {executor.submit(self.calculate_path_changes, prefix): prefix for prefix in valid_prefixes}
                
                # 收集结果
                for future in future_to_prefix:
                    try:
                        changes = future.result()
                        for key, value in changes.items():
                            all_changes[key].append(value)
                    except Exception as e:
                        log_info(f"计算路径变化时出错: {str(e)}")
        else:
            # 对于少量前缀，顺序处理
            for prefix in valid_prefixes:
                changes = self.calculate_path_changes(prefix)
                for key, value in changes.items():
                    all_changes[key].append(value)
                
        # 计算平均值
        for key in all_changes:
            if all_changes[key]:
                features[key] = sum(all_changes[key]) / len(all_changes[key])
            else:
                features[key] = 0

        return features

    def process_updates(self, updates):
        """处理更新"""
        for update in updates:
            prefix = update['prefix']
            as_path = update.get('as_path', [])
            is_withdrawal = update['type'] == 'W'
            self.update_as_info(prefix, as_path, is_withdrawal)


class BGPFeatureExtractor:
    def __init__(self, window_size=30, anomaly_start=None, anomaly_end=None):
        self.window_size = window_size
        self.anomaly_start = anomaly_start
        self.anomaly_end = anomaly_end
    
    def is_anomaly(self, timestamp):
        dt = datetime.fromtimestamp(timestamp, tz=pytz.UTC)
        return self.anomaly_start <= dt <= self.anomaly_end
        
    def extract_features_from_window(self, updates_in_window):
        """从窗口提取特征"""
        
        # 创建分析器并处理更新
        analyzer = ASPathAnalyzer()
        analyzer.process_updates(updates_in_window)
        
        # 提取AS特征
        start_time = time.time()
        as_features = analyzer.get_features()

        # 分类更新
        announcements = []
        withdrawals = []
        prefixes_a = set()
        prefixes_w = set()
        as_paths = []
        
        for u in updates_in_window:
            if u['type'] == 'A':
                announcements.append(u)
                prefixes_a.add(u['prefix'])
                if u['as_path']:
                    as_paths.append(u['as_path'])
            else:
                withdrawals.append(u)
                prefixes_w.add(u['prefix'])
        
        # 计算路径长度
        path_lengths = [len(path) for path in as_paths] if as_paths else []
        unique_paths = set(tuple(path) for path in as_paths) if as_paths else set()
        unique_path_lengths = [len(path) for path in unique_paths] if unique_paths else []
        
        # 初始化特征
        features = [0] * 25
        
        # 1-4: 基本数量特征
        features[0] = len(announcements)
        features[1] = len(withdrawals)
        features[2] = len(prefixes_a)
        features[3] = len(prefixes_w)
        
        # 5-8: AS路径特征
        if path_lengths:
            features[4] = sum(path_lengths) / len(path_lengths)
            features[5] = max(path_lengths)
            features[6] = sum(unique_path_lengths) / len(unique_path_lengths) if unique_path_lengths else 0
            features[7] = np.std(path_lengths) if len(path_lengths) > 1 else 0

        
        # 9-12: MOAS相关特征
        features[8] = as_features.get('moas_ann_num', 0)
        features[9] = as_features.get('moas_as_num', 0)
        features[10] = as_features.get('new_sub_prefix_num', 0)
        features[11] = as_features.get('moas_prefix_num', 0)
        
        # 13-22: AS路径变化特征
        features[12] = as_features.get('shorter_num', 0)
        features[13] = as_features.get('longer_num', 0)
        features[14] = as_features.get('max_shorter', 0)
        features[15] = as_features.get('max_longer', 0)
        features[16] = as_features.get('avg_shorter', 0)
        features[17] = as_features.get('avg_longer', 0)
        features[18] = as_features.get('mid_shorter', 0)
        features[19] = as_features.get('mid_longer', 0)
        features[20] = as_features.get('var_shorter', 0)
        features[21] = as_features.get('var_longer', 0)
        
        # 23-24: 前缀活跃度比率
        if announcements:
            prefix_counts = {}
            for u in announcements:
                prefix_counts[u['prefix']] = prefix_counts.get(u['prefix'], 0) + 1
            sorted_counts = sorted(prefix_counts.values(), reverse=True)
            total = len(announcements)
            
            features[22] = sorted_counts[0] / total
            features[23] = sum(sorted_counts[:3]) / total if len(sorted_counts) >= 3 else sorted_counts[0] / total
        
        # 添加标签
        last_timestamp = updates_in_window[-1]['timestamp']
        features[24] = 1 if self.is_anomaly(last_timestamp) else 0
        
        total_time = time.time() - start_time
        log_info(f"特征提取总时间: {total_time:.2f}秒")
        
        return np.array(features, dtype=np.float32)

    def process_bgp_updates(self, updates):
        """处理BGP更新，增加多线程处理窗口"""
        if not updates:
            return np.array([])
    
        # 计算时间窗口
        sorted_updates = sorted(updates, key=lambda u: u['timestamp'])
        
        # 分组到窗口
        windows = []
        current_window = []
        base_time = sorted_updates[0]['timestamp']
        
        for update in sorted_updates:
            window_num = (update['timestamp'] - base_time) // self.window_size
            if not current_window or (update['timestamp'] - base_time) // self.window_size == (current_window[0]['timestamp'] - base_time) // self.window_size:
                current_window.append(update)
            else:
                windows.append(current_window)
                current_window = [update]
        
        if current_window:
            windows.append(current_window)
        
        # 使用多线程并行处理窗口
        features_list = []
        if len(windows) > 10:
            with ThreadPoolExecutor(max_workers=8) as executor:
                # 提交任务
                future_to_window = {executor.submit(self.extract_features_from_window, window): i for i, window in enumerate(windows)}
                
                # 收集结果
                results = [None] * len(windows)
                for future in future_to_window:
                    try:
                        idx = future_to_window[future]
                        results[idx] = future.result()
                    except Exception as e:
                        log_info(f"提取特征时出错: {str(e)}")
                
                # 过滤掉None结果
                features_list = [r for r in results if r is not None]
        else:
            # 对于少量窗口，顺序处理
            for window in windows:
                if window:
                    features = self.extract_features_from_window(window)
                    features_list.append(features)
        
        # 转换为数组
        features = np.array(features_list, dtype=np.float32)
        return features


def load_bgp_updates(file_path):
    """加载BGP更新"""
    updates = []
    try:
        # 直接调用bgpdump
        cmd = ['bgpdump', '-m', file_path]
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, universal_newlines=True)
        
        # 分批处理输出以提高性能
        lines = output.splitlines()
        
        # 使用多线程处理大量行
        if len(lines) > 10000:
            # 分块处理
            chunk_size = 5000
            chunks = [lines[i:i+chunk_size] for i in range(0, len(lines), chunk_size)]
            
            # 定义处理函数
            def process_chunk(chunk):
                chunk_updates = []
                for line in chunk:
                    if line.count('|') < 5:
                        continue
                        
                    fields = line.strip().split('|')
                    msg_type = fields[2]
                    
                    # 创建更新对象
                    update = {
                        'timestamp': int(fields[1]),
                        'type': msg_type,
                        'prefix': fields[5] if len(fields) > 5 else '',
                        'as_path': [],
                        'protocol': ''
                    }
                    
                    # 跳过IPv6
                    if ':' in update['prefix']:
                        continue
                    
                    # 解析AS路径
                    if msg_type == 'A' and len(fields) > 6:
                        update['as_path'] = [asn for asn in fields[6].split() if asn.isdigit()]
                        update['protocol'] = fields[7] if len(fields) > 7 else ''
                    
                    chunk_updates.append(update)
                return chunk_updates
            
            # 使用线程池并行处理
            with ThreadPoolExecutor(max_workers=8) as executor:
                # 提交任务
                futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
                
                # 收集结果
                for future in futures:
                    try:
                        chunk_updates = future.result()
                        updates.extend(chunk_updates)
                    except Exception as e:
                        log_info(f"处理数据块时出错: {str(e)}")
        else:
            # 对于少量行，顺序处理
            for line in lines:
                if line.count('|') < 5:
                    continue
                    
                fields = line.strip().split('|')
                msg_type = fields[2]
                
                # 创建更新对象
                update = {
                    'timestamp': int(fields[1]),
                    'type': msg_type,
                    'prefix': fields[5] if len(fields) > 5 else '',
                    'as_path': [],
                    'protocol': ''
                }
                
                # 跳过IPv6
                if ':' in update['prefix']:
                    continue
                
                # 解析AS路径
                if msg_type == 'A' and len(fields) > 6:
                    update['as_path'] = [asn for asn in fields[6].split() if asn.isdigit()]
                    update['protocol'] = fields[7] if len(fields) > 7 else ''
                
                updates.append(update)
                
        return updates
        
    except Exception as e:
        log_info(f"Error loading {file_path}: {str(e)}")
        return []


def get_datetime_from_filename(filename):
    """从文件名解析日期时间
    filename格式: rrc00_updates.20200721.1410.gz
    """
    try:
        # 提取日期和时间部分
        parts = filename.split('.')
        date_str = parts[1]  # 20200721
        time_str = parts[2]  # 1410
        
        # 构建完整的日期时间字符串
        dt = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M")
        return dt.replace(tzinfo=pytz.UTC)
    except:
        return None

def filter_files_by_timerange(folder_path, start_time, end_time):
    """根据时间范围筛选文件"""
    # 计算时间差
    time_diff = end_time - start_time
    time_diff_hours = time_diff.total_seconds() / 3600
    
    # 确定实际的时间范围
    if time_diff_hours < 6:
        # 如果间隔小于6小时，扩展时间范围
        half_diff = time_diff / 2
        actual_start = start_time - half_diff
        actual_end = end_time + half_diff
    elif time_diff_hours > 6:
        # 如果间隔大于6小时，使用固定的12小时窗口
        actual_start = start_time - timedelta(hours=6)
        actual_end = start_time + timedelta(hours=6)
    else:
        # 如果正好是6小时，使用所有文件
        return sorted(folder_path.glob('*.gz'))
    
    # 筛选在时间范围内的文件
    valid_files = []
    for file_path in sorted(folder_path.glob('*.gz')):
        file_time = get_datetime_from_filename(file_path.name)
        if file_time and actual_start <= file_time <= actual_end:
            valid_files.append(file_path)
    
    return valid_files

def process_single_folder(folder, start_time, end_time, output_dir):
    """处理单个文件夹"""
    try:
        log_info(f"开始处理文件夹: {folder.name}")
        
        # 创建特征提取器
        extractor = BGPFeatureExtractor(
            window_size=30,
            anomaly_start=start_time,
            anomaly_end=end_time
        )
        
        # 根据时间范围筛选文件
        gz_files = filter_files_by_timerange(folder, start_time, end_time)
        if not gz_files:
            log_info(f"警告: 文件夹 {folder.name} 在指定时间范围内没有找到文件")
            return
            
        log_info(f"文件夹 {folder.name} 找到 {len(gz_files)} 个文件在时间范围内")
        
        # 检查输出文件
        output_file = output_dir / f"{folder.name}.txt"
        if output_file.exists():
            log_info(f"输出文件已存在且不为空，跳过文件夹: {folder.name}")
            return
            
        # 处理文件
        with open(output_file, 'w') as f:
            pass  # 创建空文件
            
        for f in gz_files:
            try:
                updates = load_bgp_updates(str(f))
                if updates:
                    features = extractor.process_bgp_updates(updates)
                    
                    # 写入特征
                    with open(output_file, 'a') as f_out:
                        for feature_vector in features:
                            formatted_features = [f"{x:.1f}" for x in feature_vector[:-1]]
                            formatted_features.append(str(int(feature_vector[-1])))
                            line = ','.join(formatted_features)
                            f_out.write(line + '\n')
                            
                # 清理内存
                del updates
                gc.collect()
                
            except Exception as e:
                log_info(f"处理文件 {f.name} 时出错: {str(e)}")
                continue
                
        log_info(f"特征已保存到 {output_file}")
        
    except Exception as e:
        log_info(f"处理文件夹 {folder.name} 时出错: {str(e)}")
        import traceback
        log_info(traceback.format_exc())

def process_all_folders():
    """处理所有文件夹"""
    # 加载事件信息
    event_info = load_event_info()
    
    # 设置路径
    data_dir = Path('/data/data/xiaolan_data/MSLSTM/some_routedata')
    output_dir = data_dir.parent / "xiaolan_data" / "MSLSTM" / "some_leak_features"
    os.makedirs(output_dir, exist_ok=True)

    # 获取文件夹
    folders = [f for f in data_dir.iterdir()]
    
    # 设置进程数
    num_processes = min(cpu_count(), 8)
    log_info(f"使用进程数: {num_processes} (CPU核心数: {cpu_count()})")
    
    # 准备参数
    process_args = []
    for folder in folders:
        event_times = event_info.get(folder.name)
        if not event_times:
            log_info(f"警告: 未找到 {folder.name} 的事件信息")
            continue
            
        if event_times['end_time'] is None:
            event_times['end_time'] = event_times['start_time'] + timedelta(hours=6)
            
        process_args.append((
            folder,
            event_times['start_time'],
            event_times['end_time'],
            output_dir
        ))
    
    # 使用进程池并行处理
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(process_single_folder, *args) for args in process_args]
        
        # 等待所有任务完成并显示进度
        completed = 0
        total = len(futures)
        for future in futures:
            try:
                future.result()
                completed += 1
                log_info(f"总进度: {completed}/{total} 个文件夹 ({completed/total*100:.1f}%)")
            except Exception as e:
                log_info(f"任务执行失败: {str(e)}")


def load_event_info():
    """加载事件信息"""
    event_info = {}
    csv_path = Path('/data/data/xiaolan_data/anomaly-event-info.csv')
    
    try:
        df = pd.read_csv(csv_path)
        log_info(f"加载事件信息: {csv_path}")
        for _, row in df.iterrows():
            event_name = row['event_name'].strip()
            
            # 处理时间
            start_time_str = row['start_time']
            start_time = pd.to_datetime(start_time_str, format='%Y/%m/%d %H:%M')
            start_time = start_time.replace(tzinfo=pytz.UTC)
            
            end_time_str = row['end_time']
            if end_time_str != 'unknown':
                end_time = pd.to_datetime(end_time_str, format='%Y/%m/%d %H:%M')
                end_time = end_time.replace(tzinfo=pytz.UTC)
            else:
                end_time = None
            
            event_info[event_name] = {
                'start_time': start_time,
                'end_time': end_time
            }
            
        
    except Exception as e:
        log_info(f"加载事件信息时出错: {str(e)}")
        return {}
        
    return event_info


if __name__ == "__main__":
    import time
    start_time = time.time()
    process_all_folders()
    end_time = time.time()