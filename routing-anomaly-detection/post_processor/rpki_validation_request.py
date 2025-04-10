#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import requests
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR/"rpki_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def rpki_valid_new(prefix, asn):
    """检查RPKI验证结果，添加错误处理和缓存文件检查"""
    cache_path = CACHE_DIR/f"{prefix}.{asn}".replace("/", "-")
    
    try:
        # 检查缓存文件是否存在且非空
        if not cache_path.exists() or cache_path.stat().st_size == 0:
            print(f"缓存文件不存在或为空: {cache_path}")
            return "unknown"
            
        # 尝试读取和解析JSON
        with open(cache_path, "r") as f:
            content = f.read().strip()
            if not content:
                print(f"缓存文件为空: {cache_path}")
                return "unknown"
            try:
                r = json.loads(content)
                return r.get("result", "unknown")
            except json.JSONDecodeError as e:
                print(f"JSON解析错误 ({cache_path}): {str(e)}")
                return "unknown"
                
    except Exception as e:
        print(f"RPKI验证出错 ({cache_path}): {str(e)}")
        return "unknown"     

def rpki_valid(prefix, asn):
    # valid, unknown, invalid_asn, invalid_length, query error
    cache_path = CACHE_DIR/f"{prefix}.{asn}".replace("/", "-")
    if cache_path.exists():
        try:
            r = json.load(open(cache_path, "r"))
        except json.decoder.JSONDecodeError as e:
            print(f"cache_path: {cache_path}")
            raise e
        return r["data"]["status"]

    payload = {"prefix": prefix, "resource": asn}
    url = "https://stat.ripe.net/data/rpki-validation/data.json"
    r = requests.get(url, params=payload)
    if r.status_code == 200:
        r = r.json()
        json.dump(r, open(cache_path, "w"))
        return r["data"]["status"]
    else:
        print(f"RPKI query error: {prefix}, {asn}")
        return "query error"
