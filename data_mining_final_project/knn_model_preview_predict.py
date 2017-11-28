import matplotlib.pyplot as plt
import inanalysis_algo.algo_component as alc
from inanalysis_algo.utils import AlgoUtils
from sklearn.datasets import load_iris
import pandas as pd


def knn_model_preview(data, predict_result, x_axis_name, y_axis_name):
    plt.scatter(data[x_axis_name], data[y_axis_name], c=predict_result, edgecolor='black', linewidth='1')
    plt.title("Iris model plot")
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    plt.savefig(x_axis_name + " _ " + y_axis_name + ".png")
    plt.close('all')


def main():
    arg_dict = {
        "n_neighbors": 5,
        "weights": "uniform"
    }
    data = load_iris()
    iris_data = pd.DataFrame(data.data, columns=data.feature_names)
    iris_label = data.target

    algo_input = alc.AlgoParam(algo_control={'mode': 'training', 'control_params': arg_dict},
                               algo_data={'data': iris_data, 'label': iris_label},
                               algo_model={'model_params': None, 'model_instance': None})
    in_algo = AlgoUtils.algo_factory('knn')
    algo_output = in_algo.do_algo(algo_input)
    model = algo_output.algo_model.model_instance
    predict_result = model.predict(iris_data)

    x_axis_name = data.feature_names[0]
    y_axis_name = data.feature_names[1]
    knn_model_preview(iris_data, predict_result, x_axis_name, y_axis_name)

if __name__ == "__main__":
    main()
