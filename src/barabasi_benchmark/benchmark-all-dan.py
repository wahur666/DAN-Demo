import csv
import multiprocessing as mp
import os

from common import timeit, load_configurations, create_demand_matrix_for_configuration
from network import OriginalDanNetwork, EgoBalanceDanNetwork, BfsDanNetwork, HuffmanDanNetwork, RandomDanNetwork

FIG_NUM = 0


@timeit
def main(show=False):
    configurations = load_configurations("../config.json")
    active_config = configurations[1]

    res = []

    vertex_nums = [25, 50, 75, 100, 125, 150, 175, 200]
    delta_nums = [10, 16, 24, 48]# , "1d", "2d", "4d", "6d", "8d", "10d", "12d"]
    constants = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    if not os.path.exists("unified_res"):
        os.mkdir("unified_res")
    res_file_original = os.path.join('unified_res', 'barabasi_unified.csv')

    # if not os.path.exists("egobalance_res"):
    #     os.mkdir("egobalance_res")
    # res_file_egobalance = os.path.join('egobalance_res', 'results_s_egobalance.csv')
    #
    # if not os.path.exists("huffman_res"):
    #     os.mkdir("huffman_res")
    # res_file_huffman = os.path.join('huffman_res', 'results_s_huffman.csv')
    #
    # if not os.path.exists("bfs_res"):
    #     os.mkdir("bfs_res")
    # res_file_bfs = os.path.join('bfs_res', 'results_s_bfs.csv')
    #
    # if not os.path.exists("random_res"):
    #     os.mkdir("random_res")
    # res_file_random = os.path.join('random_res', 'results_s_random.csv')

    fields = ['graph', 'vertex_num', 'constant', 'congestion', 'real_congestion', 'avg_route_len', 'delta',
              'max_delta', 'dan', 'most_congested_route', 'max_route_len', 'avg_tree_weight', 'most_tree_ratio',
              'tree_count', 'type']

    with open(res_file_original, 'w') as csvFile:
        writer = csv.DictWriter(csvFile, fieldnames=fields)
        writer.writeheader()
    csvFile.close()

    # with open(res_file_huffman, 'w') as csvFile:
    #     writer = csv.DictWriter(csvFile, fieldnames=fields)
    #     writer.writeheader()
    # csvFile.close()
    #
    # with open(res_file_bfs, 'w') as csvFile:
    #     writer = csv.DictWriter(csvFile, fieldnames=fields)
    #     writer.writeheader()
    # csvFile.close()
    #
    # with open(res_file_egobalance, 'w') as csvFile:
    #     writer = csv.DictWriter(csvFile, fieldnames=fields)
    #     writer.writeheader()
    # csvFile.close()
    #
    # with open(res_file_random, 'w') as csvFile:
    #     writer = csv.DictWriter(csvFile, fieldnames=fields)
    #     writer.writeheader()
    # csvFile.close()

    for vertex_num in vertex_nums:
        for delta_num in delta_nums:
            configs = []
            for constant in constants:
                for i in range(5):
                    active_cfg = active_config.copy()
                    active_cfg['vertex_num'] = vertex_num
                    active_cfg['constant'] = constant
                    active_cfg['dan'] = delta_num
                    configs.append(active_cfg)

            with mp.Pool() as p:
                res1 = p.map(run_dan_original, configs)
                res2 = p.map(run_dan_egobalance, configs)
                res3 = p.map(run_dan_huffman, configs)
                res4 = p.map(run_dan_bfs, configs)
                res5 = p.map(run_dan_random, configs)


                with open(res_file_original, "a+") as csvFile:
                    writer = csv.DictWriter(csvFile, fieldnames=fields)
                    writer.writerows(res1)
                    writer.writerows(res2)
                    writer.writerows(res3)
                    writer.writerows(res4)
                    writer.writerows(res5)
                csvFile.close()

                # with open(res_file_egobalance, "a+") as csvFile:
                #     writer = csv.DictWriter(csvFile, fieldnames=fields)
                #     writer.writerows(res2)
                # csvFile.close()
                #
                # with open(res_file_huffman, "a+") as csvFile:
                #     writer = csv.DictWriter(csvFile, fieldnames=fields)
                #     writer.writerows(res3)
                # csvFile.close()
                #
                # with open(res_file_bfs, "a+") as csvFile:
                #     writer = csv.DictWriter(csvFile, fieldnames=fields)
                #     writer.writerows(res4)
                # csvFile.close()
                #
                # with open(res_file_random, "a+") as csvFile:
                #     writer = csv.DictWriter(csvFile, fieldnames=fields)
                #     writer.writerows(res5)
                # csvFile.close()

    # if show:
    #     render_everyting(network)
    #     plt.show()


def run_dan_original(active_config):
    demand_matrix = create_demand_matrix_for_configuration(active_config)
    network = OriginalDanNetwork(demand_matrix)
    network.create_dan(active_config['dan'])
    summary = network.get_summary()
    print(active_config)
    print(summary)
    return {**summary, **active_config, "type": "original"}


def run_dan_egobalance(active_config):
    demand_matrix = create_demand_matrix_for_configuration(active_config)
    network = EgoBalanceDanNetwork(demand_matrix)
    network.create_dan(active_config['dan'])
    summary = network.get_summary()
    print(active_config)
    print(summary)
    return {**summary, **active_config, "type": "egobalance"}


def run_dan_bfs(active_config):
    demand_matrix = create_demand_matrix_for_configuration(active_config)
    network = BfsDanNetwork(demand_matrix)
    network.create_dan(active_config['dan'])
    summary = network.get_summary()
    print(active_config)
    print(summary)
    return {**summary, **active_config, "type": "bfs"}


def run_dan_huffman(active_config):
    demand_matrix = create_demand_matrix_for_configuration(active_config)
    network = HuffmanDanNetwork(demand_matrix)
    network.create_dan(active_config['dan'])
    summary = network.get_summary()
    print(active_config)
    print(summary)
    return {**summary, **active_config, "type": "huffman"}


def run_dan_random(active_config):
    demand_matrix = create_demand_matrix_for_configuration(active_config)
    network = RandomDanNetwork(demand_matrix)
    network.create_dan(active_config['dan'])
    summary = network.get_summary()
    print(active_config)
    print(summary)
    return {**summary, **active_config, "type": "random"}


if __name__ == '__main__':
    main()
