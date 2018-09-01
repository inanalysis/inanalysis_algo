import logging
from inanalysis_algo.classification.knn import Knn
from inanalysis_algo.classification.dc_tree import DCtree
from inanalysis_algo.classification.rf import RFC
from inanalysis_algo.abnormal_detection.one_class_svm import OneClassSVM
from inanalysis_algo.abnormal_detection.isolation_forest import IsolationForest
from inanalysis_algo.regression.linear_regression import LinearRegression
from inanalysis_algo.regression.bayesian_regression import BayesianRegression
from inanalysis_algo.regression.polynomial_regression import PolynomialRegression
from inanalysis_algo.clustering.kmeans import Kmeans
from inanalysis_algo.clustering.meanshift import MeanShift
from inanalysis_algo.clustering.affinitypropagation import AffinityPropagation
import enum
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class Algorithm(enum.Enum):
    one_class_svm = {
        "algo_name": "one-class_SVM",
        "project_type": "abnormal-detection"
    }
    isolation_forest = {
        "algo_name": "isolation-forest",
        "project_type": "abnormal-detection"
    }
    knn = {
        "algo_name": "knn",
        "project_type": "classification"
    }
    dc_tree = {
        "algo_name": "decision-tree",
        "project_type": "classification"
    }
    rf = {
        "algo_name": "random-forest",
        "project_type": "classification"
    }
    linear_regression = {
        "algo_name": "linear-regression",
        "project_type": "regression"
    }
    bayesian_regression = {
        "algo_name": "bayesian-regression",
        "project_type": "regression"
    }
    polynomial_regression = {
        "algo_name": "polynomial_regression",
        "project_type": "regression"
    }
    k_means = {
        "algo_name": "k-means",
        "project_type": "clustering"
    }
    meanshift = {
        "algo_name": "mean-shift",
        "project_type": "clustering"
    }
    affinity_propagation = {
        "algo_name": "affinity_propagation",
        "project_type": "clustering"
    }
    @staticmethod
    def get_project_type(algo_name):
        for algo in Algorithm:
            if algo.value['algo_name'] == algo_name:
                return algo.value['project_type']
        return None


class AlgoUtils:
    @staticmethod
    def algo_factory(model_method):
        if model_method == Algorithm.one_class_svm.value['algo_name']:
            log.debug("Abnormal-detection one-class_SVM Training")
            algo = OneClassSVM()
        elif model_method == Algorithm.isolation_forest.value['algo_name']:
            log.debug("Abnormal-detection isolation-forest Training")
            algo = IsolationForest()
        elif model_method == Algorithm.knn.value['algo_name']:
            log.debug("Classification knn Training")
            algo = Knn()
        elif model_method == Algorithm.dc_tree.value['algo_name']:
            log.debug("Classification dc-tree Training")
            algo = DCtree()
        elif model_method == Algorithm.rf.value['algo_name']:
            log.debug("Classification random-forest Training")
            algo = RFC()
        elif model_method == Algorithm.linear_regression.value['algo_name']:
            log.debug("Regression linear-regression Training")
            algo = LinearRegression()
        elif model_method == Algorithm.bayesian_regression.value['algo_name']:
            log.debug("Regression bayesian-regression Training")
            algo = BayesianRegression()
        elif model_method == Algorithm.polynomial_regression.value['algo_name']:
            log.debug("Regression polynomial_regression Training")
            algo = PolynomialRegression()
        elif model_method == Algorithm.k_means.value['algo_name']:
            log.debug("Clustering k-means Training")
            algo = Kmeans()
        elif model_method == Algorithm.meanshift.value['algo_name']:
            log.debug("Clustering mean-shift Training")
            algo = MeanShift()
        elif model_method == Algorithm.affinity_propagation.value['algo_name']:
            log.debug("Clustering affinity propagation Training")
            algo = AffinityPropagation()
        else:
            return None
        return algo
