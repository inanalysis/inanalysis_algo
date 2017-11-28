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

    def test_correct_linear_regression_parameter_type(self):
        # given: collect input parameter, create algorithm object
        arg_dict = {
            "fit_intercept": True,
            "normalize": False,
            "copy_X": True,
            "n_jobs": 1
        }
        algo_name = 'linear-regression'
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

    def test_error_linear_regression_parameter_length_not_fit_to_definition(self):
        # given: error input parameter "n_jobs" needs to be type(int)
        arg_dict = {
            "fit_intercept": True,
            "normalize": False,
            "copy_X": True,
            "n_jobs": 1.0,
            "redundant": True
        }
        algo_input = alc.AlgoParam(algo_control={'mode': 'training', 'control_params': arg_dict},
                                   algo_data={'data': self.boston_data, 'label': self.boston_label},
                                   algo_model={'model_params': None, 'model_instance': None})
        in_algo = AlgoUtils.algo_factory('linear-regression')
        input_params_definition = in_algo.get_input_params_definition()
        # when: checkout input type
        check_result = in_algo.check_input_params(input_params_definition, algo_input.algo_control.control_params)
        # then: type match
        self.assertTrue(check_result is False)

    def test_error_linear_regression_parameter_n_jobs_float_type(self):
        # given: error input parameter "n_jobs" needs to be type(int)
        arg_dict = {
            "fit_intercept": True,
            "normalize": False,
            "copy_X": True,
            "n_jobs": 1.0
        }
        algo_input = alc.AlgoParam(algo_control={'mode': 'training', 'control_params': arg_dict},
                                   algo_data={'data': self.boston_data, 'label': self.boston_label},
                                   algo_model={'model_params': None, 'model_instance': None})
        in_algo = AlgoUtils.algo_factory('linear-regression')
        input_params_definition = in_algo.get_input_params_definition()
        # when: checkout input type
        check_result = in_algo.check_input_params(input_params_definition, algo_input.algo_control.control_params)
        # then: type match
        self.assertTrue(check_result is False)

    def test_error_linear_regression_parameter_n_jobs_string_type(self):
        # given: error input parameter "n_jobs" needs to be type(int)
        arg_dict = {
            "fit_intercept": True,
            "normalize": False,
            "copy_X": True,
            "n_jobs": 'string'
        }
        algo_input = alc.AlgoParam(algo_control={'mode': 'training', 'control_params': arg_dict},
                                   algo_data={'data': self.boston_data, 'label': self.boston_label},
                                   algo_model={'model_params': None, 'model_instance': None})
        in_algo = AlgoUtils.algo_factory('linear-regression')
        input_params_definition = in_algo.get_input_params_definition()

        # when: checkout input type
        check_result = in_algo.check_input_params(input_params_definition, algo_input.algo_control.control_params)
        # then: type match
        self.assertTrue(check_result is False)

    def test_error_linear_regression_parameter_fit_intercept_string_type(self):
        # given: error input parameter "fit_intercept" needs to be type(boolean)
        arg_dict = {
            "fit_intercept": 'string',
            "normalize": False,
            "copy_X": True,
            "n_jobs": 1
        }
        algo_input = alc.AlgoParam(algo_control={'mode': 'training', 'control_params': arg_dict},
                                   algo_data={'data': self.boston_data, 'label': self.boston_label},
                                   algo_model={'model_params': None, 'model_instance': None})
        in_algo = AlgoUtils.algo_factory('linear-regression')
        input_params_definition = in_algo.get_input_params_definition()

        # when: checkout input type
        check_result = in_algo.check_input_params(input_params_definition, algo_input.algo_control.control_params)
        # then: type match
        self.assertTrue(check_result is False)

    def test_error_linear_regression_parameter_fit_intercept_int_type(self):
        # given: error input parameter "fit_intercept" needs to be type(boolean)
        arg_dict = {
            "fit_intercept": 1,
            "normalize": False,
            "copy_X": True,
            "n_jobs": 1
        }
        algo_input = alc.AlgoParam(algo_control={'mode': 'training', 'control_params': arg_dict},
                                   algo_data={'data': self.boston_data, 'label': self.boston_label},
                                   algo_model={'model_params': None, 'model_instance': None})
        in_algo = AlgoUtils.algo_factory('linear-regression')
        input_params_definition = in_algo.get_input_params_definition()

        # when: checkout input type
        check_result = in_algo.check_input_params(input_params_definition, algo_input.algo_control.control_params)
        # then: type match
        self.assertTrue(check_result is False)

    def test_error_linear_regression_parameter_fit_intercept_float_type(self):
        # given: error input parameter "fit_intercept" needs to be type(boolean)
        arg_dict = {
            "fit_intercept": 1.0,
            "normalize": False,
            "copy_X": True,
            "n_jobs": 1
        }
        algo_input = alc.AlgoParam(algo_control={'mode': 'training', 'control_params': arg_dict},
                                   algo_data={'data': self.boston_data, 'label': self.boston_label},
                                   algo_model={'model_params': None, 'model_instance': None})
        in_algo = AlgoUtils.algo_factory('linear-regression')
        input_params_definition = in_algo.get_input_params_definition()

        # when: checkout input type
        check_result = in_algo.check_input_params(input_params_definition, algo_input.algo_control.control_params)
        # then: type match
        self.assertTrue(check_result is False)

    def test_correct_linear_regression_do_algo(self):
        # given: collect input parameter, create algorithm object
        arg_dict = {
            "fit_intercept": True,
            "normalize": False,
            "copy_X": True,
            "n_jobs": 1
        }
        algo_input = alc.AlgoParam(algo_control={'mode': 'training', 'control_params': arg_dict},
                                   algo_data={'data': self.boston_data, 'label': self.boston_label},
                                   algo_model={'model_params': None, 'model_instance': None})
        in_algo = AlgoUtils.algo_factory('linear-regression')
        log.debug(algo_input)
        # when: do decision tree algorithm
        algo_output = in_algo.do_algo(algo_input)

        # then:
        self.assertTrue(algo_output is not None)
        self.assertTrue(algo_output.algo_model.model_instance is not None)

if __name__ == '__main__':
    unittest.main()
