import subprocess
import os
import gzip
import shutil
import mysql.connector
import pytricia
import ipaddress
from typing import List, Set
from datetime import datetime
from collections import defaultdict
import json

# 配置参数
BULK_INSERT_SIZE = 1000  # 批量插入数据库的记录数
BGP_DATA_FOLDER = "./hijack_"  # BGP数据文件目录

# 数据库配置
DB_CONFIG = {
    "host": "yourhost",
    "user": "yourname",
    "password": "yourpassword",
    "database": "yourdatabase"
}

def clean_as_path(as_path: str) -> List[int]:
    cleaned = []
    for part in as_path.split():
        part = part.strip("{},")  # 处理AS_SET和AS_SEQUENCE
        if part.isdigit():
            asn = int(part)
            if not cleaned or cleaned[-1] != asn:  # 去重
                cleaned.append(asn)
    return cleaned

def load_prefix_trie(cursor) -> tuple:
    pt_v4 = pytricia.PyTricia(32)
    pt_v6 = pytricia.PyTricia(128)

    cursor.execute("SELECT prefix, asn FROM prefix_as_mapping")
    for prefix, asn in cursor.fetchall():
        try:
            ip_net = ipaddress.ip_network(prefix)
             pt = pt_v4 if ip_net.version == 4 else pt_v6
            prefix_str = str(ip_net)
            if prefix_str in pt:
                if isinstance(pt[prefix_str], Set):
                    pt[prefix_str].add(asn)
                else:
                    pt[prefix_str] = {asn}
            else:
                pt[prefix_str] = {asn}
        except ValueError as e:
            print(f"[加载错误] 无效前缀 {prefix}: {e}")
    return pt_v4, pt_v6

def load_prepend_patterns_from_db(cursor):
    cursor.execute("SELECT prefix, prepend_seq FROM prepend_patterns")

    prefix_patterns = defaultdict(list)
    for prefix, path_json in cursor.fetchall():
        try:
            path = json.loads(path_json)
            if isinstance(path, list):
                prefix_patterns[prefix].append(path)
        except Exception as e:
            print(f"[解析错误] prefix={prefix}, path={path_json}: {e}")
    return prefix_patterns

def detect_type_0_hijack(
    prefix: str,
    origin_asn: int,
    pt_v4: pytricia.PyTricia,
    pt_v6: pytricia.PyTricia
) -> bool:
    try:
        ip_net = ipaddress.ip_network(prefix)
        pt = pt_v4 if ip_net.version == 4 else pt_v6
        prefix_str = str(ip_net)

        if prefix_str in pt:
            return origin_asn not in pt[prefix_str]  # AS不匹配则判定为劫持
        return False  # 前缀未注册时不在此检测

    except Exception as e:
        print(f"[Type0检测错误] {prefix}: {e}")
        return False
        
def detect_type_1_hijack(
    as_path: List[int],
    cursor
) -> bool:
    if len(as_path) < 2:
        return False  # 单AS路径不检测

    first_hop = as_path[0]  # 第一跳AS
    origin_as = as_path[-1]  # 起源AS

    # 查询AS关系（适配 provider-customer|-1 和 peer|peer|0 格式）
    cursor.execute(
        """SELECT relationship FROM as_relationship
                WHERE ((asn1 = %s AND asn2 = %s AND relationship = -1) OR  # provider-customer
              (asn1 = %s AND asn2 = %s AND relationship = -1) OR     # 反向provider-customer
              (asn1 = %s AND asn2 = %s AND relationship = 0)) # peer关系
        LIMIT 1""",
        (first_hop, origin_as, origin_as, first_hop, first_hop, origin_as)
    )

    # 无结果表示无任何合法关系
    return cursor.fetchone() is None

def detect_type_p_hijack(as_path: list, prepend_patterns: list) -> tuple[int, str]:
    if not as_path or not prepend_patterns:
        return -1, "-"

    orig_path = as_path
    best_match_len = 0
    hijacker = -1
    
    for pattern in prepend_patterns:
        if len(orig_path) < len(pattern) + 1:
            continue
        segment = orig_path[-len(pattern)-1:-1]  # 从尾部排除 origin

        if segment == pattern:
            return -1, "-"  # 匹配合法尾部，合法路径

        # 否则计算匹配长度
        match_len = 0
        for a, b in zip(segment[::-1], pattern[::-1]):
            if a == b:
                match_len += 1
            else:
                break

        if match_len > best_match_len:
            best_match_len = match_len
            try:
                hijacker = orig_path[-match_len - 2]
            except IndexError:
                hijacker = -1

    if hijacker != -1:
        return hijacker, "P"
    return -1, "-"

def detect_prefix_squatting_or_subprefix(
    prefix: str,
    origin_asn: int,
    pt_v4: pytricia.PyTricia,
    pt_v6: pytricia.PyTricia
) -> tuple:
    try:
        ip_net = ipaddress.ip_network(prefix)
        pt = pt_v4 if ip_net.version == 4 else pt_v6
        prefix_str = str(ip_net)

        if prefix_str in pt:
            return ("REGISTERED", None)

        supernet = pt.get_key(prefix_str)
        if supernet:
            return ("SUBPREFIX", supernet)
            
        return ("SQUATTING", None)
    except ValueError:
        return ("ERROR", None)

def parse_bgp_mrt_file(mrt_file_path: str, pt_v4, pt_v6, prepend_dict, db_cursor, conn):
    hijack_records = []

    try:
        result = subprocess.run(
            f"bgpdump -m {mrt_file_path} | grep '|A|'",
            shell=True,
            capture_output=True,
            text=True
        )

        for line in result.stdout.splitlines():
            parts = line.split("|")
            if len(parts) < 7:
                continue

            prefix = parts[5].strip()
            as_path = parts[6].strip()
            timestamp = datetime.utcfromtimestamp(int(parts[1].strip()))
            try:
                cleaned_as_path = clean_as_path(as_path)
                if not cleaned_as_path:
                    continue
                origin_asn = cleaned_as_path[-1]

                # Step 1: 子前缀/Squatting 检测
                prefix_type, _ = detect_prefix_squatting_or_subprefix(
                    prefix, origin_asn, pt_v4, pt_v6
                )
                if prefix_type == "SUBPREFIX":
                    hijack_records.append((prefix, origin_asn, "Subprefix Hijack"), timestamp)
                    continue
                elif prefix_type == "SQUATTING":
                    hijack_records.append((prefix, origin_asn, "Prefix Squatting", timestamp))
                    continue

                # Step 2: Type-0 Hijack
                if detect_type_0_hijack(prefix, origin_asn, pt_v4, pt_v6):
                    hijack_records.append((prefix, origin_asn, "Origin Hijack", timestamp))
                    continue

                # Step 3: Type-1 Hijack
                if len(cleaned_as_path) >= 2:
                    is_type1 = detect_type_1_hijack(cleaned_as_path, db_cursor)
                      if is_type1:
                        hijack_records.append((prefix, origin_asn, "Path Fabrication Hijack", timestamp))
                        continue

                patterns = prepend_dict.get(prefix, [])
                hijacker, p_type = detect_type_p_hijack(cleaned_as_path, patterns)
                if p_type == "P":
                    hijack_records.append((prefix, hijacker, "Type-P Hijack"), timestamp)
                    continue

                # 批量插入逻辑
                if len(hijack_records) >= BULK_INSERT_SIZE:
                    db_cursor.executemany(
                        "INSERT INTO hijack_events (prefix, origin_asn, hijack_type, timestamp) "
                        "VALUES (%s, %s, %s, %s)",
                        hijack_records
                    )
                    conn.commit()
                    hijack_records = []

            except Exception as e:
                print(f"处理错误: {line[:50]}... | 错误: {e}")
        # 插入剩余记录
        if hijack_records:
            db_cursor.executemany(
                "INSERT INTO hijack_events (prefix, origin_asn, hijack_type, timestamp) "
                "VALUES (%s, %s, %s, %s)",
                hijack_records
            )
            conn.commit()

    except Exception as e:
        print(f"[文件解析异常] {mrt_file_path}: {e}")

def parse_all_gz_files(
    folder_path: str,
    pt_v4: pytricia.PyTricia,
    pt_v6: pytricia.PyTricia,
    prepend_dict: dict,
    db_cursor,
    conn
):
    """
    处理目录下所有.gz压缩的MRT文件
    """
    for file_name in sorted(os.listdir(folder_path)):
        if not file_name.endswith(".gz"):
            continue

        gz_path = os.path.join(folder_path, file_name)
        mrt_path = gz_path[:-3]  # 去除.gz后缀

        try:
            # 解压文件
            with gzip.open(gz_path, 'rb') as f_in, open(mrt_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

            # 解析MRT文件
            parse_bgp_mrt_file(mrt_path, pt_v4, pt_v6, prepend_dict, db_cursor, conn)

        except Exception as e:
            print(f"[处理异常] {file_name}: {e}")
        finally:
            # 清理临时文件
            if os.path.exists(mrt_path):
                os.remove(mrt_path)

def main():
    # 连接数据库
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(buffered=True)

    try:
        # 加载前缀-AS映射数据
        pt_v4, pt_v6 = load_prefix_trie(cursor)
        print(f"已加载 {len(pt_v4)} 条IPv4前缀和 {len(pt_v6)} 条IPv6前缀")

        # 处理BGP数据文件
        prepend_dict = load_prepend_patterns_from_db(cursor)
        parse_all_gz_files(BGP_DATA_FOLDER, pt_v4, pt_v6, prepend_dict, cursor, conn)

    except Exception as e:
        print(f"[主程序错误] {e}")
    finally:
        cursor.close()
        conn.close()
        print("数据库连接已关闭")

if __name__ == "__main__":
    main()
             
