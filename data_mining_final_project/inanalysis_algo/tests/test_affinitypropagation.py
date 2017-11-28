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

    def test_correct_ap_parameter_type(self):
        # given: collect input parameter, create algorithm object
        arg_dict = {
            "damping": 0.5,
            "preference": "-50",
            "convergence_iter": 15,
            "max_iter": 200
        }
        algo_name = 'affinity_propagation'
        algo_input = alc.AlgoParam(algo_control={'mode': 'training', 'control_params': arg_dict},
                                   algo_data={'data': self.iris_data, 'label': None},
                                   algo_model={'model_params': None, 'model_instance': None})
        in_algo = AlgoUtils.algo_factory(algo_name)
        input_params_definition = in_algo.get_input_params_definition()
        # when: checkout input type
        check_result = in_algo.check_input_params(input_params_definition, algo_input.algo_control.control_params)
        # then: type match
        self.assertTrue(check_result is True)
        self.assertEqual(Algorithm.get_project_type(algo_name), "clustering")

    def test_correct_affinitypropagation_do_algo(self):
        # given: collect input parameter, create algorithm object
        arg_dict = {
            "damping": 0.5,
            "preference": "-50",
            "convergence_iter": 15,
            "max_iter": 200
        }
        algo_name = 'affinity_propagation'
        algo_input = alc.AlgoParam(algo_control={'mode': 'training', 'control_params': arg_dict},
                                   algo_data={'data': self.iris_data, 'label': None},
                                   algo_model={'model_params': None, 'model_instance': None})
        in_algo = AlgoUtils.algo_factory(algo_name)
        log.debug(algo_input)

        # when: do decision tree algorithm
        algo_output = in_algo.do_algo(algo_input)

        # then:
        self.assertTrue(algo_output is not None)
        self.assertTrue(algo_output.algo_model.model_instance is not None)

    def test_error_ap_damping_as_string(self):
        # given: collect input parameter, create algorithm object
        arg_dict = {
            "damping": "hi",
            "preference": "-50",
            "convergence_iter": 15,
            "max_iter": 200
        }
        algo_name = 'affinity_propagation'
        algo_input = alc.AlgoParam(algo_control={'mode': 'training', 'control_params': arg_dict},
                                   algo_data={'data': self.iris_data, 'label': None},
                                   algo_model={'model_params': None, 'model_instance': None})
        in_algo = AlgoUtils.algo_factory(algo_name)
        input_params_definition = in_algo.get_input_params_definition()
        # when: checkout input type
        check_result = in_algo.check_input_params(input_params_definition, algo_input.algo_control.control_params)
        # then: type match
        self.assertTrue(check_result is False)

    def test_error_ap_damping_out_of_range(self):
        # given: collect input parameter, create algorithm object
        arg_dict = {
            "damping": 1.5,
            "preference": "-50",
            "convergence_iter": 15,
            "max_iter": 200
        }
        algo_name = 'affinity_propagation'
        algo_input = alc.AlgoParam(algo_control={'mode': 'training', 'control_params': arg_dict},
                                   algo_data={'data': self.iris_data, 'label': None},
                                   algo_model={'model_params': None, 'model_instance': None})
        in_algo = AlgoUtils.algo_factory(algo_name)
        input_params_definition = in_algo.get_input_params_definition()
        # when: checkout input type
        check_result = in_algo.check_input_params(input_params_definition, algo_input.algo_control.control_params)
        # then: type match
        self.assertTrue(check_result is False)

    def test_error_ap_conv_iter_as_float(self):
        # given: collect input parameter, create algorithm object
        arg_dict = {
            "damping": 0.5,
            "preference": "-50",
            "convergence_iter": 7.5,
            "max_iter": 200
        }
        algo_name = 'affinity_propagation'
        algo_input = alc.AlgoParam(algo_control={'mode': 'training', 'control_params': arg_dict},
                                   algo_data={'data': self.iris_data, 'label': None},
                                   algo_model={'model_params': None, 'model_instance': None})
        in_algo = AlgoUtils.algo_factory(algo_name)
        input_params_definition = in_algo.get_input_params_definition()
        # when: checkout input type
        check_result = in_algo.check_input_params(input_params_definition, algo_input.algo_control.control_params)
        # then: type match
        self.assertTrue(check_result is False)

    def test_error_ap_max_iter_as_float(self):
        # given: collect input parameter, create algorithm object
        arg_dict = {
            "damping": 0.5,
            "preference": "-50",
            "convergence_iter": 7.5,
            "max_iter": 1.5
        }
        algo_name = 'affinity_propagation'
        algo_input = alc.AlgoParam(algo_control={'mode': 'training', 'control_params': arg_dict},
                                   algo_data={'data': self.iris_data, 'label': None},
                                   algo_model={'model_params': None, 'model_instance': None})
        in_algo = AlgoUtils.algo_factory(algo_name)
        input_params_definition = in_algo.get_input_params_definition()
        # when: checkout input type
        check_result = in_algo.check_input_params(input_params_definition, algo_input.algo_control.control_params)
        # then: type match
        self.assertTrue(check_result is False)

if __name__ == '__main__':
    unittest.main()
