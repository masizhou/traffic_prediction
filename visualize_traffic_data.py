import numpy as np
import matplotlib.pyplot as plt


def get_flow(file_name):
    flow_data = np.load(file_name)
    flow_data = flow_data['data'].transpose([1, 0, 2])[:, :, 0][:, :, np.newaxis]  # [N, T, D]  D = 1

    return flow_data


if __name__ == '__main__':
    traffic_data = get_flow("PeMS_04/PeMS04.npz")
    node_id = 226
    print(traffic_data.shape)

    plt.plot(traffic_data[: 24 * 12, node_id, 0])
    plt.savefig("node_{:3d}_1.png".format(node_id))

    plt.plot(traffic_data[: 24 * 12, node_id, 1])
    plt.savefig("node_{:3d}_2.png".format(node_id))

    plt.plot(traffic_data[: 24 * 12, node_id, 2])
    plt.savefig("node_{:3d}_3.png".format(node_id))
