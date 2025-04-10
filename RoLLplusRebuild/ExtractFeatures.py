import sys
import logging
logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S', format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)
from ASFeature import ASFeature

asFeature2021 = ASFeature(rel_filename='./data/20210601.as-rel2.txt', org_filename='./data/20211001.as-org2info.txt', 
                          pfx2as_filename="mydata_prefix2as_data/CAIDA/routeviews-rv2-20210601-1200.pfx2as", feature_root='./features')
asFeature2022 = ASFeature(rel_filename='./data/20220601.as-rel2.txt', org_filename='./data/20221001.as-org2info.txt', 
                          pfx2as_filename="mydata_prefix2as_data/CAIDA/routeviews-rv2-20220601-1200.pfx2as", feature_root='./features')
asFeature2023 = ASFeature(rel_filename='./data/20231001.as-rel2.txt', org_filename='./data/20231001.as-org2info.txt', 
                          pfx2as_filename="mydata_prefix2as_data/CAIDA/routeviews-rv2-20231001-1200.pfx2as", feature_root='./features')

with open('allsample.csv', 'r') as fin, open('reGenerateSamples.csv', 'w') as fout:
    for line in fin:
        origin_data = line.split(',')
        year = int(origin_data[0])
        Features:ASFeature = getattr(sys.modules[__name__], f'asFeature{year}')
        if Features == None:
            exit(0)
        triplet = origin_data[1:4]
        # triplet = [int(x) for x in triplet]
        label = origin_data[-1].strip()
        

        asDistance0 = Features.getASDistance(triplet[0])
        asDistance1 = Features.getASDistance(triplet[1])
        asDistance2 = Features.getASDistance(triplet[2])
        asDegree0 = Features.getASDegree(triplet[0])
        asDegree1 = Features.getASDegree(triplet[1])
        asDegree2 = Features.getASDegree(triplet[2])
        asAddressSpace0 = Features.getASAddressSpace(triplet[0])
        asAddressSpace1 = Features.getASAddressSpace(triplet[1])
        asAddressSpace2 = Features.getASAddressSpace(triplet[2])
        asCountry = Features.getTripletCountry(triplet)
        asRIR = Features.getTripletRIR(triplet)
        asType0 = Features.getASType(triplet[0])
        asType1 = Features.getASType(triplet[1])
        asType2 = Features.getASType(triplet[2])
        asBetweennessCentrality0 = Features.getASBetweennessCentrality(triplet[0])
        asBetweennessCentrality1 = Features.getASBetweennessCentrality(triplet[1])
        asBetweennessCentrality2 = Features.getASBetweennessCentrality(triplet[2])
        asClosenessCentrality0 = Features.getASClosenessCentrality(triplet[0])
        asClosenessCentrality1 = Features.getASClosenessCentrality(triplet[1])
        asClosenessCentrality2 = Features.getASClosenessCentrality(triplet[2])
        asEigenvectorCentrality0 = Features.getASEigenvectorCentrality(triplet[0])
        asEigenvectorCentrality1 = Features.getASEigenvectorCentrality(triplet[1])
        asEigenvectorCentrality2 = Features.getASEigenvectorCentrality(triplet[2])
        asClusteringCoefficient0 = Features.getASClusteringCoefficient(triplet[0])
        asClusteringCoefficient1 = Features.getASClusteringCoefficient(triplet[1])
        asClusteringCoefficient2 = Features.getASClusteringCoefficient(triplet[2])
        asSquareClustering0 = Features.getASSquareClustering(triplet[0])
        asSquareClustering1 = Features.getASSquareClustering(triplet[1])
        asSquareClustering2 = Features.getASSquareClustering(triplet[2])
        # asRouterNumber0 = Features.getASRouterNumber(triplet[0])
        # asRouterNumber1 = Features.getASRouterNumber(triplet[1])
        # asRouterNumber2 = Features.getASRouterNumber(triplet[2])
        asAverageNeighborDegree0 = Features.getASAverageNeighborDegree(triplet[0])
        asAverageNeighborDegree1 = Features.getASAverageNeighborDegree(triplet[1])
        asAverageNeighborDegree2 = Features.getASAverageNeighborDegree(triplet[2])
        # asMaxCliqueSize0 = Features.getASMaxCliqueSize(triplet[0])
        # asMaxCliqueSize1 = Features.getASMaxCliqueSize(triplet[1])
        # asMaxCliqueSize2 = Features.getASMaxCliqueSize(triplet[2])
        asTrianglesClustering0 = Features.getASTrianglesClustering(triplet[0])
        asTrianglesClustering1 = Features.getASTrianglesClustering(triplet[1])
        asTrianglesClustering2 = Features.getASTrianglesClustering(triplet[2])
        a=[
            year, triplet[0], triplet[1], triplet[2],
            asDistance0, asDistance1, asDistance2,
            asDegree0, asDegree1, asDegree2,
            asAddressSpace0, asAddressSpace1, asAddressSpace2,
            asCountry, asRIR,
            asType0, asType1, asType2,
            asBetweennessCentrality0, asBetweennessCentrality1, asBetweennessCentrality2,
            asClosenessCentrality0, asClosenessCentrality1, asClosenessCentrality2,
            asEigenvectorCentrality0, asEigenvectorCentrality1, asEigenvectorCentrality2,
            asClusteringCoefficient0, asClusteringCoefficient1, asClusteringCoefficient2,
            asSquareClustering0, asSquareClustering1, asSquareClustering2,
            # asRouterNumber0, asRouterNumber1, asRouterNumber2,
            asAverageNeighborDegree0, asAverageNeighborDegree1, asAverageNeighborDegree2,
            # asMaxCliqueSize0, asMaxCliqueSize1, asMaxCliqueSize2,
            asTrianglesClustering0, asTrianglesClustering1, asTrianglesClustering2,
            label
        ]

        a = [str(x) for x in a]
        s=','.join(a)+'\n'
        if '-1000' in s:
            log.error(f'triplet: {str(triplet)}\nfeature:{s}')
            continue
        fout.write(s)
