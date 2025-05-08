#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from pathlib import Path
from io import StringIO
import pandas as pd
import subprocess
from monitor import Monitor
import zipfile

bgpd_dir = Path(__file__).resolve().parent.parent / "data" / "routeviews"
dataset_dir = Path('/data/data/anomaly-event-routedata')

def extract(file_name):
    zip_path = dataset_dir / f"{file_name}.zip"
    extract_dir = zip_path.parent

    # 解压操作
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"解压成功！文件保存在: {extract_dir}")
        extract_file_dir = extract_dir / file_name
        return extract_file_dir
    except FileNotFoundError:
        print(f"错误: ZIP 文件未找到: {zip_path}")
    except zipfile.BadZipFile:
        print(f"错误: 文件损坏或不是有效的 ZIP 文件: {zip_path}")

def load_updates_to_df(fpath, bgpd=bgpd_dir/"bgpd"):
    res = subprocess.check_output([str(bgpd), "-q", "-m", "-u", str(fpath)]).decode()
    fmt = "type|timestamp|A/W|peer-ip|peer-asn|prefix|as-path|origin-protocol|next-hop|local-pref|MED|community|atomic-agg|aggregator|unknown-field-1|unknown-field-2"
    cols = fmt.split("|")
    
    processed_lines = []
    for line in res.strip().split('\n'):
        if '|A|' in line:  
            processed_lines.append(line)
    processed_data = '\n'.join(processed_lines)
    
    df = pd.read_csv(StringIO(processed_data), sep="|", names=cols, usecols=cols[:-2], dtype=str, keep_default_na=False)
    return df

def detect(data, route_change_dir):
    mon = Monitor()

    for fpath in data:
        print(fpath)
        _, date, time, _ = fpath.name.split(".")

        df = load_updates_to_df(fpath)
        df = df.sort_values(by="timestamp")

        mon.consume(df, detect=True)

        route_change_df = pd.DataFrame.from_records(mon.route_changes)
        mon.route_changes = []

        route_change_df.to_csv(route_change_dir/f"{date}.{time}.csv", index=False)

def detect_for_all_files():
    all_files = [f.stem for f in dataset_dir.glob("*.zip")]
    print(f"找到以下文件需要处理：")
    for file in all_files:
        print(f"- {file}")
    print("="*50)

    for file_name in all_files:
        print(f"\n开始处理文件: {file_name}")
        print("-"*30)
        
        result_dir = Path('/data/data/xiaolan_data/beam/detection_result/ripe/route_change')
        route_change_dir = result_dir / file_name
        route_change_dir.mkdir(exist_ok=True, parents=True)
        
        extract_dir = dataset_dir / file_name
        if extract_dir.exists():
            print(f"文件 {file_name} 已解压，直接处理")
            file_paths = list(extract_dir.glob("*.gz"))
        else:
            print(f"开始解压文件: {file_name}")
            extract_file_dir = extract(file_name)
            file_paths = list(extract_file_dir.glob("*.gz"))
        
        if not file_paths:
            print(f"警告: {file_name} 解压后没有找到文件")
            continue
            
        print(f"开始检测路由变化...")
        detect(file_paths, route_change_dir)
        print(f"文件 {file_name} 处理完成")
        print("="*50)

if __name__ == "__main__":
    detect_for_all_files()
