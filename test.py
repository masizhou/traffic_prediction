import matplotlib.pyplot as plt
import numpy as np
import h5py

def visualize_result(h5_file, nodes_id, time_se, visualize_file):
    file_obj = h5py.File(h5_file, "r")
    print('这个file_obj:',  file_obj["predict"][:][:, :, 0].shape)
    prediction = file_obj["predict"][:][:, :, 0]  # [N, T]
    print('预测：', prediction.shape, prediction)
    target = file_obj["target"][:][:, :, 0]  # [N, T]
    print('目标：', target)
    file_obj.close()

    # plot_prediction = prediction[nodes_id][time_se[0]: time_se[1]]  # [T1]
    # plot_target = target[nodes_id][time_se[0]: time_se[1]]  # [T1]
    #
    # plt.figure()
    # plt.grid(True, linestyle="-.", linewidth=0.5)
    # plt.plot(np.array([t for t in range(time_se[1] - time_se[0])]), plot_prediction, ls="-", marker=" ", color="r")
    # plt.plot(np.array([t for t in range(time_se[1] - time_se[0])]), plot_target, ls="-", marker=" ", color="b")
    #
    # plt.legend(["prediction", "target"], loc="upper right")
    #
    # plt.axis([0, time_se[1] - time_se[0],
    #           np.min(np.array([np.min(plot_prediction), np.min(plot_target)])),
    #           np.max(np.array([np.max(plot_prediction), np.max(plot_target)]))])
    #
    # plt.savefig(visualize_file + ".png")


if __name__ == '__main__':
    visualize_result(h5_file="GAT_result.h5",
                     nodes_id=120,
                     time_se=[0, 24 * 12 * 2],
                     visualize_file="gat_node_120")