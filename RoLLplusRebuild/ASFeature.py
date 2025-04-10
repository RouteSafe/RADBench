import os
import networkx as nx
import json
import ipaddress
import logging
import datetime
from zoneinfo import ZoneInfo # py3.9后是系统库
def beijing(sec, what):
    beijing_time = datetime.datetime.now(ZoneInfo('Asia/Shanghai')) # 返回北京时间
    return beijing_time.timetuple()
logging.Formatter.converter = beijing
logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S', format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

class ASFeature:
    def __init__(self, rel_filename, org_filename, pfx2as_filename, feature_root='./mydata_features', astype_filename='mydata_astype_data/20210401.as2types.txt'):
        """管理ASFeature

        Args:
            rel_filename (str): as rel 文件路径
            org_filename (str): as org 文件路径
            pfx2as_filename (str): pfx2as 文件路径
            feature_root (str): 计算后特征的保存路径
            astype_filename (str, optional): astype文件路径. Defaults to 'mydata_astype_data/20210401.as2types.txt'.
        """
        if rel_filename == None or org_filename == None:
            return
        self.rel_filename = rel_filename
        self.org_filename = org_filename
        self.pfx2as_filename = pfx2as_filename
        self.feature_root = feature_root
        self.astype_filename = astype_filename
        if 'as-rel2' in self.rel_filename:
            self.G = nx.read_edgelist(rel_filename, nodetype=int, data=(("weight", int),("source", str),), comments='#', delimiter="|")
        else:
            self.G = nx.read_edgelist(rel_filename, nodetype=int, data=(("weight", int),), comments='#', delimiter="|")
        # as rel 相关
        self.ixpList = []
        self.cliqueList = []  # tier-1 as集合
        self.asSet = set()  # as集合
        self.asLinkDict = {}  # as与它的邻居
        self.t1AsShortestLength = {} # tier-1 as到各as的最短距离

        # as org相关
        self.as2OrgDict = {}
        self.rir2Idx = {}

        self.as2Prefix4 = {}
        self.as2Prefix6 = {}
        self.as2AddressSpace = {}

        # 需要的数据
        self.asDistance = {}
        self.asDegree = {}
        # self.asLocation = {}  # self.as2OrgDict
        self.asType = {} 
        self.asBetweennessCentrality = {}
        self.asClosenessCentrality = {}
        self.asEigenvectorCentrality = {}
        self.asClusteringCoefficient = {}
        self.asSquareClustering = {}
        self.asRouterNum = {}  # todo ?

        self.AverageNeighborDegree = {}
        self.asMaxCliqueSize = {}
        self.asTrianglesClustering = {}

        self.org2As = {}
        self.SiblingAs = []

        # 读取rel文件
        log.info(f"读取rel文件: {self.rel_filename}")
        with open(rel_filename) as f:
            for line in f:
                line = line.strip()
                if "clique" in line:
                    tmp=line.split(": ")[1].split(" ")
                    self.cliqueList = [int(x) for x in tmp]
                    continue
                if "IXP" in line:
                    tmp=line.split(": ")[1].split(" ")
                    self.ixpList = [int(x) for x in tmp]
                    continue
                if line.startswith("#"):
                    continue

                tmp=line.split("|")
                self.asSet.add(int(tmp[0]))
                self.asSet.add(int(tmp[1]))
                self.asLinkDict.setdefault(int(tmp[0]),set()).add(int(tmp[1]))
                self.asLinkDict.setdefault(int(tmp[1]),set()).add(int(tmp[0]))

                # 计算degree
                self.asDegree.setdefault(int(tmp[0]), 0)
                self.asDegree[int(tmp[0])] += 1
                self.asDegree.setdefault(int(tmp[1]), 0)
                self.asDegree[int(tmp[1])] += 1


        # 读取org
        log.info(f"读取org文件: {self.org_filename}")
        with open(org_filename) as f:
            org_detail = {}
            flag = -1
            for line in f:
                line = line.strip()
                if "# format:org_id|changed|org_name|country|source" in line:
                    flag=0
                    continue
                elif "# format:aut|changed|aut_name|org_id" in line:
                    # 两种格式
                    # format:aut|changed|aut_name|org_id|source
                    # format:aut|changed|aut_name|org_id|opaque_id|source
                    flag=1
                    continue
                if flag == 0:
                    org_id, changed, org_name, country, source = line.split("|")
                    # todo:特殊情况：有多个RIR的怎么算？
                    if source=="JPNIC":
                        source = 'APNIC'
                    elif source=="APNIC,RIPE" and country=='HK':
                        source = 'APNIC'
                    elif source=="LACNIC,RIPE,APNIC" and country=='CN' and org_id=='@family-8228':
                        source = 'APNIC'
                    org_detail[org_id] = {
                        'org_name': org_name,
                        'country': country,
                        'rir': source,
                    }
                elif flag == 1:
                    tmp = line.split("|")
                    aut = tmp[0]
                    org_id = tmp[3]
                    source = tmp[-1]
                    # aut, changed, aut_name, org_id, opaque_id, source = line.split("|")
                    self.as2OrgDict[aut] = org_detail[org_id]
                    self.as2OrgDict[aut]['org_id'] = org_id
                    self.org2As.setdefault(org_id, []).append(aut)
        # 计算sibling as
        log.info(f"计算sibling as")
        for k, v in self.org2As.items():
            if len(v)>1:
                self.SiblingAs.append(v)

        # 读取astype
        log.info(f"读取astype文件: {self.astype_filename}")
        types = ['Content', 'Enterprise', 'Transit/Access']
        with open(self.astype_filename, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                tmp = line.strip().split('|')
                self.asType[tmp[0]] = types.index(tmp[2])

        # 读取whois文件
        log.info(f"读取pfx2as文件: {self.pfx2as_filename}")
        with open(self.pfx2as_filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line=='':
                    continue
                prefix, prefix_1, asn = line.split('\t')
                prefix = f'{prefix}/{prefix_1}'
                self.as2Prefix4.setdefault(asn, []).append(prefix)

        riswhoisfiles = os.listdir('mydata_prefix2as_data/RIPE')
        riswhois_filename = sorted(riswhoisfiles)[0]
        log.info(f"读取riswhois文件: {riswhois_filename}")
        log.info("todo?: 合并riswhois和pfx2as的前缀")
        with open(os.path.join('mydata_prefix2as_data/RIPE', riswhois_filename), 'r') as f:
            for line in f:
                if '%' in line or line.strip()=='':
                    continue
                asn, prefix, _ = line.strip().split('\t')
                if prefix == '0.0.0.0/0':
                    continue
                if asn not in self.as2Prefix4.keys():
                    self.as2Prefix4.setdefault(asn, []).append(prefix)
        # with open('riswhoisdump.IPv6', 'r') as f:
        #     for line in f:
        #         if '%' in line or line.strip()=='':
        #             continue
        #         asn, prefix, _ = line.strip().split('\t')
        #         if prefix == '::/0':
        #             continue
        #         self.as2Prefix6.setdefault(asn, []).append(prefix)
        # 计算as地址空间大小
        self.calcAsAddressSpace()

        self.calcASDistance()
        self.calcASBetweennessCentrality()
        self.calcASClosenessCentrality()
        self.calcASEigenvectorCentrality()
        self.calcAsClusteringCoefficient()
        self.calcAsSquareClustering()

        self.calcASAverageNeighborDegree()
        # self.calcASMaxCliqueSize()
        self.calcASTrianglesClustering()

    # 大约40min
    def calcASDistance(self, ):
        # 检查本地是否已有计算结果
        date = os.path.basename(self.rel_filename).split('.')[0]
        filename = f'{self.feature_root}/{date}/{date}-ASDistance.json'
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                self.asDistance = json.load(f)
            log.info(f'ASDistance: 读取本地数据{filename}')
            return

        log.info(f'ASDistance: 正在计算并保存至{filename}')
        self.t1AsShortestLength = {}
        def bfs(src):
            visited = []
            queue = []
            visited.append(src)
            queue.append(src)
            
            while queue:
                s = queue.pop(0)
                #弹出列表的第一个item
                #自己到自己的路径长度是1

                # print (s, end = " ") 
                for neighbour in self.asLinkDict[s]:
                    if neighbour not in visited:
                        visited.append(neighbour)
                        queue.append(neighbour)
                        self.t1AsShortestLength[(src,neighbour)]=self.t1AsShortestLength[(src,s)]+1

        # bfs计算tier-1 as到各as的距离
        for clique_item in self.cliqueList:
            self.t1AsShortestLength[(clique_item,clique_item)]=1
            bfs(clique_item)
        # 计算as到tier-1 as距离，取平均值
        for x in self.asSet:
            sum=0
            for c in self.cliqueList:
                if (c, x) in self.t1AsShortestLength:
                    sum += self.t1AsShortestLength[(c, x)]
                else:
                    # 不可达
                    sum += 1000
            self.asDistance[x] = sum/len(self.cliqueList)
        
        # 保存计算结果
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(self.asDistance, f, indent=4)
        log.info(f'ASDistance: 计算结束')

    # 大约12h
    def calcASBetweennessCentrality(self, ):
        # 检查本地是否已有计算结果
        date = os.path.basename(self.rel_filename).split('.')[0]
        filename = f'{self.feature_root}/{date}/{date}-ASBetweennessCentrality.json'
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                self.asBetweennessCentrality = json.load(f)
            log.info(f'ASBetweennessCentrality: 读取本地数据{filename}')
            return
        
        log.info(f'ASBetweennessCentrality: 正在计算并保存至{filename}')
        self.asBetweennessCentrality = nx.betweenness_centrality(self.G)

        # 保存计算结果
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(self.asBetweennessCentrality, f, indent=4)
        log.info(f'ASBetweennessCentrality: 计算结束')

    # 大约3h
    def calcASClosenessCentrality(self, ):
        # 检查本地是否已有计算结果
        date = os.path.basename(self.rel_filename).split('.')[0]
        filename = f'{self.feature_root}/{date}/{date}-ASClosenessCentrality.json'
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                self.asClosenessCentrality = json.load(f)
            log.info(f'ASClosenessCentrality: 读取本地数据{filename}')
            return
        
        log.info(f'ASClosenessCentrality: 正在计算并保存至{filename}')
        self.asClosenessCentrality = nx.closeness_centrality(self.G)

        # 保存计算结果
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(self.asClosenessCentrality, f, indent=4)
        log.info(f'ASClosenessCentrality: 计算结束')
    

    def calcASEigenvectorCentrality(self, ):
        # 检查本地是否已有计算结果
        date = os.path.basename(self.rel_filename).split('.')[0]
        filename = f'{self.feature_root}/{date}/{date}-ASEigenvectorCentrality.json'
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                self.asEigenvectorCentrality = json.load(f)
            log.info(f'ASEigenvectorCentrality: 读取本地数据{filename}')
            return
        
        log.info(f'ASEigenvectorCentrality: 正在计算并保存至{filename}')
        self.asEigenvectorCentrality = nx.eigenvector_centrality(self.G)

        # 保存计算结果
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(self.asEigenvectorCentrality, f, indent=4)
        log.info(f'ASEigenvectorCentrality: 计算结束')


    # 大约1min
    def calcAsClusteringCoefficient(self, ):
        # 检查本地是否已有计算结果
        date = os.path.basename(self.rel_filename).split('.')[0]
        filename = f'{self.feature_root}/{date}/{date}-AsClusteringCoefficient.json'
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                self.asClusteringCoefficient = json.load(f)
            log.info(f'AsClusteringCoefficient: 读取本地数据{filename}')
            return
        
        log.info(f'AsClusteringCoefficient: 正在计算并保存至{filename}')
        
        self.asClusteringCoefficient = nx.clustering(self.G)

        # 保存计算结果
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(self.asClusteringCoefficient, f, indent=4)
        log.info(f'ASClusteringCoefficient: 计算结束')


    # 大约4h
    def calcAsSquareClustering(self, ):
        # 检查本地是否已有计算结果
        date = os.path.basename(self.rel_filename).split('.')[0]
        filename = f'{self.feature_root}/{date}/{date}-AsSquareClustering.json'
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                self.asSquareClustering = json.load(f)
            log.info(f'AsSquareClustering: 读取本地数据{filename}')
            return
        
        log.info(f'AsSquareClustering: 正在计算并保存至{filename}')
        
        self.asSquareClustering = nx.square_clustering(self.G)

        # 保存计算结果
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(self.asSquareClustering, f, indent=4)
        log.info(f'AsSquareClustering: 计算结束')

    
    def calcASAverageNeighborDegree(self, ):
        # 检查本地是否已有计算结果
        date = os.path.basename(self.rel_filename).split('.')[0]
        filename = f'{self.feature_root}/{date}/{date}-AsAverageNeighborDegree.json'
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                self.asAverageNeighborDegree = json.load(f)
            log.info(f'AsAverageNeighborDegree: 读取本地数据{filename}')
            return
        
        log.info(f'AsAverageNeighborDegree: 正在计算并保存至{filename}')
        self.asAverageNeighborDegree = nx.average_neighbor_degree(self.G)

        # 保存计算结果
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(self.asAverageNeighborDegree, f, indent=4)
        log.info(f'AsAverageNeighborDegree: 计算结束')

    # 
    def calcASMaxCliqueSize(self, ):
        # 检查本地是否已有计算结果
        date = os.path.basename(self.rel_filename).split('.')[0]
        filename = f'{self.feature_root}/{date}/{date}-AsMaxCliqueSize.json'
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                self.asMaxCliqueSize = json.load(f)
            log.info(f'AsMaxCliqueSize: 读取本地数据{filename}')
            return
        
        log.info(f'AsMaxCliqueSize: 正在计算并保存至{filename}')
        self.asMaxCliqueSize = nx.node_clique_number(self.G)

        # 保存计算结果
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(self.asMaxCliqueSize, f, indent=4)
        log.info(f'AsMaxCliqueSize: 计算结束')

    # 
    def calcASTrianglesClustering(self, ):
        # 检查本地是否已有计算结果
        date = os.path.basename(self.rel_filename).split('.')[0]
        filename = f'{self.feature_root}/{date}/{date}-AsTrianglesClustering.json'
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                self.asTrianglesClustering = json.load(f)
            log.info(f'AsTrianglesClustering: 读取本地数据{filename}')
            return
        
        log.info(f'AsTrianglesClustering: 正在计算并保存至{filename}')
        self.asTrianglesClustering = nx.triangles(self.G)

        # 保存计算结果
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(self.asTrianglesClustering, f, indent=4)
        log.info(f'AsTrianglesClustering: 计算结束')

    def calcAsAddressSpace(self, ):
        log.info(f'正在计算AS地址空间')
        for asn in self.as2Prefix4.keys():
            sum = 0
            for prefix in self.as2Prefix4[asn]:
                # network = ipaddress.ip_network(prefix, strict=False)
                # sum += network.num_addresses
                _, prefix_len = prefix.split('/')
                sum += 2 ** (32 - int(prefix_len)) - 2
            # for prefix in self.as2Prefix6[asn]:
            #     network = ipaddress.ip_network(prefix, strict=False)
            #     sum += network.num_addresses
            #     _, prefix_len = prefix.split('/')
            #     sum += 2 ** (128 - int(prefix_len)) - 2
            self.as2AddressSpace[asn] = sum
    
    def getASDistance(self, asn:str):
        return self.asDistance.get(asn, -1000)

    def getASDegree(self, asn:str):
        return self.asDegree.get(int(asn), -1000)
    
    def getASAddressSpace(self, asn:str):
        return self.as2AddressSpace.get(asn, -1000)
        # todo 
        if asn not in self.as2Prefix4.keys():
            return -1000
        sum = 0
        for prefix in self.as2Prefix4[asn]:
            # network = ipaddress.ip_network(prefix, strict=False)
            # sum += network.num_addresses
            _, prefix_len = prefix.split('/')
            sum += 2 ** (32 - int(prefix_len)) - 2
        # if asn not in self.as2Prefix6.keys() and sum==0:
        #     return -1000
        # for prefix in self.as2Prefix6[asn]:
        #     network = ipaddress.ip_network(prefix, strict=False)
        #     sum += network.num_addresses
        #     _, prefix_len = prefix.split('/')
        #     sum += 2 ** (128 - int(prefix_len)) - 2
        return sum

    def getTripletCountry(self, triplet:list[str]):
        for t in triplet:
            if t not in self.as2OrgDict.keys():
                return -1000
        a1, a2, a3 = triplet
        if self.as2OrgDict[a1]['country'] == self.as2OrgDict[a2]['country'] and self.as2OrgDict[a1]['country'] != self.as2OrgDict[a3]['country']:
            return 1
        elif self.as2OrgDict[a1]['country'] == self.as2OrgDict[a3]['country'] and self.as2OrgDict[a1]['country'] != self.as2OrgDict[a2]['country']:
            return 2
        elif self.as2OrgDict[a2]['country'] == self.as2OrgDict[a3]['country'] and self.as2OrgDict[a1]['country'] != self.as2OrgDict[a3]['country']:
            return 3
        elif self.as2OrgDict[a1]['country'] != self.as2OrgDict[a2]['country'] and self.as2OrgDict[a1]['country'] != self.as2OrgDict[a3]['country']:
            return 4
        elif self.as2OrgDict[a1]['country'] == self.as2OrgDict[a2]['country'] and self.as2OrgDict[a1]['country'] == self.as2OrgDict[a3]['country']:
            return 5
        else:
            return -1000
        
    
    def getTripletRIR(self, triplet:list[str]):
        for t in triplet:
            if t not in self.as2OrgDict.keys():
                return -1000
        rir0 = self.as2OrgDict[triplet[0]]['rir']
        rir1 = self.as2OrgDict[triplet[1]]['rir']
        rir2 = self.as2OrgDict[triplet[2]]['rir']
        if rir0 == rir1 and rir0 != rir2:
            return 1
        elif rir0 == rir2 and rir0 != rir1:
            return 2
        elif rir1 == rir2 and rir0 != rir2:
            return 3
        elif rir0 != rir1 and rir0 != rir2:
            return 4
        elif rir0 == rir1 and rir0 == rir2:
            return 5
        else:
            return -1000
    
    def getASType(self, asn:str):
        return self.asType.get(asn, -1000)
    
    def getASBetweennessCentrality(self, asn:str):
        return self.asBetweennessCentrality.get(asn, -1000)
    
    def getASClosenessCentrality(self, asn:str):
        return self.asClosenessCentrality.get(asn, -1000)
    
    def getASEigenvectorCentrality(self, asn:str):
        return self.asEigenvectorCentrality.get(asn, -1000)
    
    def getASClusteringCoefficient(self, asn:str):
        return self.asClusteringCoefficient.get(asn, -1000)
    
    def getASSquareClustering(self, asn:str):
        return self.asSquareClustering.get(asn, -1000)
    
    def getASAverageNeighborDegree(self, asn:str):
        return self.asAverageNeighborDegree.get(asn, -1000)

    def getASMaxCliqueSize(self, asn:str):
        return self.asMaxCliqueSize.get(asn, -1000)
    
    def getASTrianglesClustering(self, asn:str):
        return self.asTrianglesClustering.get(asn, -1000)
    
    def getASRouterNumber(self, asn:str):
        # todo
        return 0
    
    def updateAsRel(rel_filename):
        pass


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='计算as的各种特征')
    parser.add_argument('--rel_filename', type=str)
    parser.add_argument('--org_filename', type=str)
    args = parser.parse_args()

    # asTopology = ASFeature(rel_filename='./data/20231001.as-rel2.txt', org_filename='./data/20231001.as-org2info.txt')
    asTopology = ASFeature(rel_filename=args.rel_filename, org_filename=args.org_filename, pfx2as_filename="",
                            feature_root='./data', astype_filename='mydata_astype_data/20210401.as2types.txt')
    # asTopology.readGraph()
    # bc = nx.betweenness_centrality(asTopology.G)
    pass