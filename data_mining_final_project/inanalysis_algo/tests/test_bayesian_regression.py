import unittest
import inanalysis_algo.algo_component as alc
from inanalysis_algo.utils import AlgoUtils
from inanalysis_algo.utils import Algorithm
from sklearn.datasets import load_boston
import logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class InAlgoTestCase(unittest.TestCase):

    def setUp(self):
        data = load_boston()
        self.boston_data = data.data
        self.boston_label = data.target

    def tearDown(self):
        del self.boston_data
        del self.boston_label

    def test_correct_bayesian_regression_parameter_type(self):
        # given: collect input parameter, create algorithm object
        arg_dict = {
            "n_iter": 300,
            "tol": 0.001
        }
        algo_name = 'bayesian-regression'
        algo_input = alc.AlgoParam(algo_control={'mode': 'training', 'control_params': arg_dict},
                                   algo_data={'data': self.boston_data, 'label': self.boston_label},
                                   algo_model={'model_params': None, 'model_instance': None})
        in_algo = AlgoUtils.algo_factory(algo_name)
        input_params_definition = in_algo.get_input_params_definition()
        # when: checkout input type
        check_result = in_algo.check_input_params(input_params_definition, algo_input.algo_control.control_params)
        # then: type match
        self.assertTrue(check_result is True)
        self.assertEqual(Algorithm.get_project_type(algo_name), "regression")

    def test_error_bayesian_regression_parameter_length_not_fit_to_definition(self):
        # given: error number of input parameters more
        arg_dict = {
            "n_iter": 300,
            "tol": 0.001,
            "normalize": True
        }
        algo_input = alc.AlgoParam(algo_control={'mode': 'training', 'control_params': arg_dict},
                                   algo_data={'data': self.boston_data, 'label': self.boston_label},
                                   algo_model={'model_params': None, 'model_instance': None})
        in_algo = AlgoUtils.algo_factory('bayesian-regression')
        input_params_definition = in_algo.get_input_params_definition()
        # when: checkout input type
        check_result = in_algo.check_input_params(input_params_definition, algo_input.algo_control.control_params)
        # then: type match
        self.assertTrue(check_result is False)

    def test_error_bayesian_regression_parameter_n_iter_float_type(self):
        # given: error input parameter "n_jobs" needs to be type(int)
        arg_dict = {
            "n_iter": 300.0,
            "tol": 0.001
        }
        algo_input = alc.AlgoParam(algo_control={'mode': 'training', 'control_params': arg_dict},
                                   algo_data={'data': self.boston_data, 'label': self.boston_label},
                                   algo_model={'model_params': None, 'model_instance': None})
        in_algo = AlgoUtils.algo_factory('bayesian-regression')
        input_params_definition = in_algo.get_input_params_definition()
        # when: checkout input type
        check_result = in_algo.check_input_params(input_params_definition, algo_input.algo_control.control_params)
        # then: type match
        self.assertTrue(check_result is False)

    def test_error_bayesian_regression_parameter_n_iter_string_type(self):
        # given: error input parameter "n_iter" needs to be type(int)
        arg_dict = {
            "n_iter": 'string',
            "tol": 0.0001
        }
        algo_input = alc.AlgoParam(algo_control={'mode': 'training', 'control_params': arg_dict},
                                   algo_data={'data': self.boston_data, 'label': self.boston_label},
                                   algo_model={'model_params': None, 'model_instance': None})
        in_algo = AlgoUtils.algo_factory('bayesian-regression')
        input_params_definition = in_algo.get_input_params_definition()

        # when: checkout input type
        check_result = in_algo.check_input_params(input_params_definition, algo_input.algo_control.control_params)
        # then: type match
        self.assertTrue(check_result is False)

    def test_error_bayesian_regression_parameter_tol_boolean_type(self):
        # given: error input parameter "tol" needs to be type(float)
        arg_dict = {
            "n_iter": 'string',
            "tol": True
        }
        algo_input = alc.AlgoParam(algo_control={'mode': 'training', 'control_params': arg_dict},
                                   algo_data={'data': self.boston_data, 'label': self.boston_label},
                                   algo_model={'model_params': None, 'model_instance': None})
        in_algo = AlgoUtils.algo_factory('bayesian-regression')
        input_params_definition = in_algo.get_input_params_definition()

        # when: checkout input type
        check_result = in_algo.check_input_params(input_params_definition, algo_input.algo_control.control_params)
        # then: type match
        self.assertTrue(check_result is False)

    def test_correct_bayesian_regression_do_algo(self):
        # given: collect input parameter, create algorithm object
        arg_dict = {
            "n_iter": 300,
            "tol": 0.001
        }
        algo_input = alc.AlgoParam(algo_control={'mode': 'training', 'control_params': arg_dict},
                                   algo_data={'data': self.boston_data, 'label': self.boston_label},
                                   algo_model={'model_params': None, 'model_instance': None})
        in_algo = AlgoUtils.algo_factory('bayesian-regression')
        log.debug(algo_input)
        # when: do decision tree algorithm
        algo_output = in_algo.do_algo(algo_input)

        # then:
        self.assertTrue(algo_output is not None)
        self.assertTrue(algo_output.algo_model.model_instance is not None)

if __name__ == '__main__':
    unittest.main()
