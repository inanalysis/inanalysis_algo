import unittest
import inanalysis_algo.algo_component as alc
from inanalysis_algo.utils import AlgoUtils
from inanalysis_algo.utils import Algorithm
from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
import logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class InAlgoTestCase(unittest.TestCase):

    def setUp(self):
        data = load_iris()
        self.iris_data = data.data
        self.iris_label = data.target
        data = load_boston()
        self.boston_data = data.data
        self.boston_label = data.target

    def tearDown(self):
        del self.iris_data
        del self.iris_label
        del self.boston_data
        del self.boston_label

    def test_correct_one_class_svm_parameter_gamma_auto_type(self):
        # given: collect input parameter, create algorithm object
        arg_dict = {
            "gamma": 'auto',
            "nu": 0.5,
            "kernel": "rbf",
            "degree": 3
        }
        algo_name ='one-class_SVM'
        algo_input = alc.AlgoParam(algo_control={'mode': 'training', 'control_params': arg_dict},
                                   algo_data={'data': self.iris_data, 'label': None},
                                   algo_model={'model_params': None, 'model_instance': None})
        in_algo = AlgoUtils.algo_factory(algo_name)
        input_params_definition = in_algo.get_input_params_definition()
        # when: checkout input type
        check_result = in_algo.check_input_params(input_params_definition, algo_input.algo_control.control_params)
        # then: type match
        self.assertTrue(check_result is True)
        self.assertEqual(Algorithm.get_project_type(algo_name), "abnormal-detection")

    def test_correct_one_class_svm_parameter_gamma_float_type(self):
        # given: collect input parameter, create algorithm object
        arg_dict = {
            "gamma": 0.1,
            "nu": 0.5,
            "kernel": "rbf",
            "degree": 3
        }
        algo_name ='one-class_SVM'
        algo_input = alc.AlgoParam(algo_control={'mode': 'training', 'control_params': arg_dict},
                                   algo_data={'data': self.iris_data, 'label': None},
                                   algo_model={'model_params': None, 'model_instance': None})
        in_algo = AlgoUtils.algo_factory(algo_name)
        input_params_definition = in_algo.get_input_params_definition()
        # when: checkout input type
        check_result = in_algo.check_input_params(input_params_definition, algo_input.algo_control.control_params)
        # then: type match
        self.assertTrue(check_result is True)
        self.assertEqual(Algorithm.get_project_type(algo_name), "abnormal-detection")

    def test_correct_one_class_svm_do_algo(self):
        # given: collect input parameter, create algorithm object
        arg_dict = {
            "gamma": 0.5,
            "nu": 0.5,
            "kernel": "rbf",
            "degree": 3
        }
        algo_input = alc.AlgoParam(algo_control={'mode': 'training', 'control_params': arg_dict},
                                   algo_data={'data': self.iris_data, 'label': None},
                                   algo_model={'model_params': None, 'model_instance': None})
        in_algo = AlgoUtils.algo_factory('one-class_SVM')
        log.debug(algo_input)

        # when: do decision tree algorithm
        algo_output = in_algo.do_algo(algo_input)

        # then:
        self.assertTrue(algo_output is not None)
        self.assertTrue(algo_output.algo_model.model_instance is not None)

    def test_error_one_class_svm_parameter_gamma_string_type(self):
        # given: collect input parameter, create algorithm object
        arg_dict = {
            "gamma": "string",
            "nu": 0.5,
            "kernel": "rbf",
            "degree": 3
        }
        algo_input = alc.AlgoParam(algo_control={'mode': 'training', 'control_params': arg_dict},
                                   algo_data={'data': self.iris_data, 'label': None},
                                   algo_model={'model_params': None, 'model_instance': None})
        in_algo = AlgoUtils.algo_factory('one-class_SVM')
        input_params_definition = in_algo.get_input_params_definition()
        # when: checkout input type
        check_result = in_algo.check_input_params(input_params_definition, algo_input.algo_control.control_params)
        # then: type match
        self.assertTrue(check_result is False)

    def test_error_one_class_svm_do_algo_wrong_data_input(self):
        # given: collect input parameter, create algorithm object
        arg_dict = {
            "gamma": 0.5,
            "nu": 0.5,
            "kernel": "rbf",
            "degree": 3
        }
        # None data input
        algo_input = alc.AlgoParam(algo_control={'mode': 'training', 'control_params': arg_dict},
                                   algo_data={'data': None, 'label': None},
                                   algo_model={'model_params': None, 'model_instance': None})
        in_algo = AlgoUtils.algo_factory('one-class_SVM')
        log.debug(algo_input)

        # when: do decision tree algorithm
        algo_output = in_algo.do_algo(algo_input)

        # then:
        self.assertTrue(algo_output is None)

if __name__ == '__main__':
    unittest.main()
