import matplotlib.pyplot as plt
import sys

from common import load_configurations, create_demand_matrix_for_configuration, render_everyting
from network import EgoBalanceDanNetwork, Network

def main(show=False):
    configurations = load_configurations()
    # active_config = configurations[0]

    active_config = configurations[5]

    demand_matrix = create_demand_matrix_for_configuration(active_config)

    network = EgoBalanceDanNetwork(demand_matrix)

    network.create_dan(active_config['dan'])

    if show:
        render_everyting(network)
        plt.show()


if __name__ == '__main__':
    render = True if len(sys.argv) == 2 and sys.argv[1] == "-r" else False
    main(render)
