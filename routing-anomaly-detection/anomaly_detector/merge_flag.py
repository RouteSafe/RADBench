import pandas as pd
from pathlib import Path
import logging
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from datetime import datetime
import pytz

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def calculate_metrics(df, start_time, end_time):
    """计算评估指标"""
    try:
        # 确保timestamp列已经是datetime格式
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)
        
        # 添加UTC时区
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize(pytz.UTC)
        if start_time.tz is None:
            start_time = start_time.tz_localize(pytz.UTC)
        if end_time.tz is None:
            end_time = end_time.tz_localize(pytz.UTC)
            
        # 向下取整到分钟
        df['timestamp'] = df['timestamp'].dt.floor('min')
        start_time = start_time.floor('min')
        end_time = end_time.floor('min')
        
        logging.info(f"Timestamp dtype: {df['timestamp'].dtype}")
        logging.info(f"Start time: {start_time}")
        logging.info(f"End time: {end_time}")
        
        # 定义实际异常事件标签
        df['true_label'] = df['timestamp'].apply(
            lambda x: 1 if start_time <= x <= end_time else 0
        )

        # 获取真实标签和预测标签
        y_true = df['true_label']
        y_pred = df['type'].astype(int)

        # 计算混淆矩阵
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # 计算告警数量（预测为1的数量）
        alarm_count = int(y_pred.sum())

        # 计算各项指标，包括AUC和告警数量
        metrics = {
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'fp_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'fn_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
            'auc': roc_auc_score(y_true, y_pred) if len(set(y_true)) > 1 else 0,
            'alarm_count': alarm_count  # 添加告警数量
        }
        
        return metrics
        
    except Exception as e:
        logging.error(f"计算指标时出错: {str(e)}")
        logging.error(f"DataFrame head: {df.head()}")
        logging.error(f"DataFrame info: {df.info()}")
        raise

def get_event_type_and_column(event_id):
    """根据事件ID确定事件类型和需要保留的列"""
    event_type = event_id.split('-')[0].lower()
    if event_type == 'hijack':
        return 'hijack', 'prefix_hijacking'
    elif event_type == 'leak':
        return 'leak', 'route_leakage'
    elif event_type == 'outage':
        return 'outage', 'outage'
    else:
        raise ValueError(f"未知的事件类型: {event_type}")


def merge_flags(flag_dir, output_file, event_info):
    """合并标记并计算指标"""
    # 读取所有 CSV 文件并合并
    flag_files = list(Path(flag_dir).rglob("*.csv"))
    if not flag_files:
        logging.warning(f"目录 {flag_dir} 中没有找到CSV文件")
        return
        
    all_flags_data = []
    
    # 从文件夹名称获取事件ID和对应的列名
    event_id = flag_dir.name.split('.')[0]  # 去掉.flags后缀
    try:
        event_type, column_name = get_event_type_and_column(event_id)
        logging.info(f"事件 {event_id} 类型为 {event_type}，将保留 {column_name} 列")
    except ValueError as e:
        logging.error(str(e))
        return
    
    for file in flag_files:
        try:
            df = pd.read_csv(file)
            if column_name not in df.columns:
                logging.error(f"文件 {file} 中未找到列 {column_name}")
                continue
            
            try:
                # 将Unix时间戳转换为datetime格式并精确到分钟
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s').dt.floor('min')
                
                # 添加UTC时区
                df["timestamp"] = df["timestamp"].dt.tz_localize(pytz.UTC)
                
                all_flags_data.append(df[["timestamp", column_name]])
                
            except Exception as e:
                logging.error(f"处理文件 {file} 的timestamp时出错: {str(e)}")
                continue
        except Exception as e:
            logging.error(f"处理文件 {file} 时出错: {str(e)}")
            continue

    if not all_flags_data:
        logging.warning(f"没有成功读取任何数据")
        return
        
    merged_flags = pd.concat(all_flags_data, ignore_index=True)
    
    # 按时间排序并重置索引
    merged_flags = merged_flags.sort_values("timestamp").reset_index(drop=True)
    
    # 直接保存结果
    merged_flags.to_csv(output_file, index=False)
    logging.info(f"合并后的 flag 文件已保存为: {output_file}")
    
    # 计算评估指标
    try:
        # 从event_info中获取对应的start_time和end_time
        event_row = event_info[event_info['event_name'] == event_id].iloc[0]
        start_time = pd.to_datetime(event_row['start_time'])
        end_time = pd.to_datetime(event_row['end_time'])
        
        # 重命名列以适配评估函数
        merged_flags = merged_flags.rename(columns={column_name: 'type'})
        metrics = calculate_metrics(merged_flags, start_time, end_time)
        
        metrics['event_id'] = event_id
        metrics['event_type'] = event_type
        
        logging.info(f"事件 {event_id} 的评估结果: {metrics}")
        return metrics
        
    except Exception as e:
        logging.error(f"计算评估指标时出错: {str(e)}")
        return None

def process_all_flag_folders():
    base_dir = Path('/data/data/xiaolan_data/beam/detection_result/ripe/reported_alarms')
    
    # 读取异常事件信息
    event_info_path = Path('/data/data/xiaolan_data/anomaly-event-info.csv')
    try:
        event_info = pd.read_csv(event_info_path)
        # 转换start_time
        event_info['start_time'] = pd.to_datetime(event_info['start_time'])
        
        # 处理end_time，如果是unknown则设置为start_time + 6小时
        def process_end_time(row):
            if row['end_time'] == 'unknown':
                return row['start_time'] + pd.Timedelta(hours=6)
            return pd.to_datetime(row['end_time'])
            
        event_info['end_time'] = event_info.apply(process_end_time, axis=1)
        
        logging.info("成功读取异常事件信息文件")
    except Exception as e:
        logging.error(f"读取异常事件信息文件失败: {str(e)}")
        return
    
    # 获取所有以.flags结尾的文件夹
    flag_folders = [f for f in base_dir.iterdir() if f.is_dir() and f.name.endswith('.flags')]
    
    if not flag_folders:
        logging.warning("没有找到.flags结尾的文件夹")
        return
        
    logging.info(f"找到 {len(flag_folders)} 个.flags文件夹需要处理")
    
    # 创建存储所有评估结果的列表
    all_metrics = []
    
    for folder in flag_folders:
        try:
            # 创建输出目录
            output_dir = base_dir / "merged_flags"
            output_dir.mkdir(exist_ok=True)
            
            # 生成输出文件名（去掉.flags后缀）
            output_file = output_dir / f"{folder.name[:-11]}_merged.csv"
            
            logging.info(f"处理文件夹: {folder.name}")
            metrics = merge_flags(folder, output_file, event_info)
            
            if metrics is not None:
                all_metrics.append(metrics)
            
        except Exception as e:
            logging.error(f"处理文件夹 {folder.name} 时出错: {str(e)}")
            continue
    
    # 保存所有评估结果到一个CSV文件
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_output = base_dir / "merged_flags" / "all_metrics_withauc.csv"
        metrics_df.to_csv(metrics_output, index=False)
        logging.info(f"所有评估指标已保存到: {metrics_output}")
        
        # 按事件类型计算指定列的平均指标，包括AUC和告警数量
        columns_to_average = ['precision', 'recall', 'f1', 'fp_rate', 'fn_rate', 'auc', 'alarm_count']
        avg_metrics = metrics_df.groupby('event_type')[columns_to_average].mean()
        avg_output = base_dir / "merged_flags" / "average_metrics_withauc.csv"
        avg_metrics.to_csv(avg_output)
        logging.info(f"平均评估指标已保存到: {avg_output}")
        
    logging.info("所有文件夹处理完成")

if __name__ == "__main__":
    process_all_flag_folders()