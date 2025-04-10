#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from datetime import datetime
from pathlib import Path
import pandas as pd
import os

from utils import load_emb_distance

def get_model_for_date(event_date):
    """根据事件日期选择对应的模型"""
    date_ranges = {
        # 前两个模型使用 rel1
        ('20041201', 'rel'): (datetime(2004, 12, 1), datetime(2015, 1, 1)),
        ('20150101', 'rel'): (datetime(2015, 1, 1), datetime(2016, 5, 1)),
        # 后三个模型使用 rel2
        ('20160501', 'rel2'): (datetime(2016, 5, 1), datetime(2021, 11, 1)),
        ('20211101', 'rel2'): (datetime(2021, 11, 1), datetime(2024, 8, 1)),
        ('20240801', 'rel2'): (datetime(2024, 8, 1), datetime(2025, 1, 1))  # 添加最新的模型
    }
    
    for (model_date, rel_version), (start_date, end_date) in date_ranges.items():
        if start_date <= event_date < end_date:
            return f"{model_date}.as-{rel_version}.1000.10.128"
    
    return None

def evaluate_test_for(file_name):
    # 从文件名解析事件日期
    try:
        # 假设文件名格式为 "type-YYYYMMDD-description"
        date_str = file_name.split('-')[1]
        event_date = datetime.strptime(date_str, '%Y%m%d')
    except Exception as e:
        print(f"无法从文件名 {file_name} 解析日期: {str(e)}")
        return
    
    # 获取对应的模型
    beam_model = get_model_for_date(event_date)
    if not beam_model:
        print(f"未找到适用于日期 {event_date.strftime('%Y-%m-%d')} 的模型")
        return
        
    print(f"\n处理文件夹: {file_name}")
    print(f"事件日期: {event_date.strftime('%Y-%m-%d')}")
    print(f"使用模型: {beam_model}")
    print("-"*30)
    
    repo_dir = Path(__file__).resolve().parent.parent
    model_dir = repo_dir/"BEAM_engine"/"models"
    collector_result_dir = Path('/data/data/xiaolan_data/beam/detection_result/ripe')
    route_change_dir = collector_result_dir/"route_change"/file_name
    beam_metric_dir = collector_result_dir/"BEAM_metric"/file_name
    beam_metric_dir.mkdir(exist_ok=True, parents=True)

    emb_dir = model_dir/beam_model
    emb_d, dtw_d, path_d, emb, _, _ = load_emb_distance(emb_dir, return_emb=True)

    def dtw_d_only_exist(s, t):
        return dtw_d([i for i in s if i in emb], [i for i in t if i in emb])

    for i in os.listdir(route_change_dir):
        file_path = Path(route_change_dir) / i
        beam_metric_file = Path(beam_metric_dir) / f"{file_path.stem}.bm.csv"
        if beam_metric_file.exists(): 
            continue

        df = pd.read_csv(file_path)

        path1 = [s.split(" ") for s in df["path1"].values]
        path2 = [t.split(" ") for t in df["path2"].values]

        metrics = pd.DataFrame.from_dict({
            "diff": [dtw_d(s,t) for s,t in zip(path1, path2)], 
            "diff_only_exist": [dtw_d_only_exist(s,t) for s,t in zip(path1, path2)], 
            "path_d1": [path_d(i) for i in path1],
            "path_d2": [path_d(i) for i in path2],
            "path_l1": [len(i) for i in path1],
            "path_l2": [len(i) for i in path2],
            "head_tail_d1": [emb_d(i[0], i[-1]) for i in path1],
            "head_tail_d2": [emb_d(i[0], i[-1]) for i in path2],
        })
        
        metrics.to_csv(beam_metric_file, index=False)

    print(f"文件夹 {file_name} 处理完成")
    print("="*50)

def process_all_folders():
    repo_dir = Path(__file__).resolve().parent.parent
    collector_result_dir = Path('/data/data/xiaolan_data/beam/detection_result/ripe')
    route_change_base_dir = collector_result_dir/"route_change"
    
    folders = [f.name for f in route_change_base_dir.iterdir() if f.is_dir()]
    
    print(f"找到以下文件夹需要处理：")
    for folder in folders:
        print(f"- {folder}")
    print("="*50)

    for folder in folders:
        evaluate_test_for(folder)

if __name__ == "__main__":
    process_all_folders()

