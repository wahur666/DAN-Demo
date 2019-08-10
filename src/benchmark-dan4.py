import csv
import multiprocessing as mp
import os

from common import timeit, load_configurations, create_demand_matrix_for_configuration
from network import HuffmanDanNetwork

FIG_NUM = 0

@timeit
def main(show=False):
    configurations = load_configurations()
    active_config = configurations[0]

    res = []

    vertex_nums = [25, 50, 75, 100, 125, 150, 175, 200]
    delta_nums = [10, 16, 24, 48, "1d", "2d", "4d", "6d", "8d", "10d", "12d"]
    constants = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    if not os.path.exists("huffman_res"):
        os.mkdir("huffman_res")
    res_file = os.path.join('huffman_res', 'results_huffman.csv')

    fields = ['graph', 'vertex_num', 'constant', 'congestion', 'real_congestion', 'avg_route_len', 'delta',
              'max_delta', 'dan', 'most_congested_route', 'type']

    with open(res_file, 'w') as csvFile:
        writer = csv.DictWriter(csvFile, fieldnames=fields)
        writer.writeheader()
    csvFile.close()

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
                res = p.map(run_dan, configs)

                with open(res_file, "a+") as csvFile:
                    writer = csv.DictWriter(csvFile, fieldnames=fields)
                    #writer.writeheader()
                    writer.writerows(res)

                csvFile.close()

    # if show:
    #     render_everyting(network)
    #     plt.show()


def run_dan(active_config):
    demand_matrix = create_demand_matrix_for_configuration(active_config)
    network = HuffmanDanNetwork(demand_matrix)
    network.create_dan(active_config['dan'])
    summary = network.get_summary()
    print(active_config)
    print(summary)
    return {**summary, **active_config, "type": "huffman"}


if __name__ == '__main__':
    main()

