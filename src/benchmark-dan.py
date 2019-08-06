import csv
import multiprocessing as mp
import os

from math import ceil

from common import timeit, load_configurations, create_demand_matrix_for_configuration
from network import OriginalDanNetwork

FIG_NUM = 0


@timeit
def main(show=False):
    configurations = load_configurations()
    active_config = configurations[0]

    res = []

    vertex_nums = [25, 50, 75, 100, 125, 150, 175, 200]
    delta_nums = [10, 16, 24, 48, "1d", "2d", "4d", "6d", "8d", "10d", "12d"]
    ratios = [0.25, 0.33]

    if not os.path.exists("original_res"):
        os.mkdir("original_res")
    res_file = os.path.join('original_res', 'results.csv')
    with open(res_file, 'w') as csvFile:
        fields = ['graph', 'vertex_num', 'constant', 'congestion', 'real_congestion', 'avg_route_len', 'delta',
                  'max_delta', 'dan', 'most_congested_route', 'ratio', 'type']
        writer = csv.DictWriter(csvFile, fieldnames=fields, dialect='excel', delimiter=';')
        writer.writeheader()
    csvFile.close()

    for vertex_num in vertex_nums:
        for delta_num in delta_nums:
            configs = []
            for ratio in ratios:
                for i in range(5):
                    active_cfg = active_config.copy()
                    active_cfg['vertex_num'] = vertex_num
                    active_cfg['constant'] = int(ceil(vertex_num * ratio))
                    active_cfg['dan'] = delta_num
                    active_cfg['ratio'] = ratio
                    configs.append(active_cfg)

            with mp.Pool() as p:
                res = p.map(run_dan, configs)

                with open(res_file, "a+") as csvFile:
                    fields = ['graph', 'vertex_num', 'constant', 'congestion','real_congestion', 'avg_route_len', 'delta', 'max_delta', 'dan', 'most_congested_route', 'ratio', 'type']
                    writer = csv.DictWriter(csvFile, fieldnames=fields, dialect='excel', delimiter=';')
                    #writer.writeheader()
                    writer.writerows(res)

                csvFile.close()

    # if show:
    #     render_everyting(network)
    #     plt.show()


def run_dan(active_config):
    demand_matrix = create_demand_matrix_for_configuration(active_config)
    network = OriginalDanNetwork(demand_matrix)
    network.create_dan(active_config['dan'])
    summary = network.get_summary()
    print(active_config)
    print(summary)
    return {**summary, **active_config, "ratio": active_config["ratio"], "type": "original"}



if __name__ == '__main__':
    main()
