from sklearn import linear_model
import inanalysis_algo.algo_component as alc
import logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class ParamsDefinitionSet(alc.ParamsDefinitionSet):
    def __init__(self):
        self.params_definition_set =\
            {
                alc.ParamsDefinition(name='n_iter', type='int', range='', default_value='300', description='Maximum number of iterations'),
                alc.ParamsDefinition(name='tol', type='float', range='', default_value='0.001', description='Stop the algorithm if converged below this')
            }


class BayesianRegression(alc.InanalysisAlgo):
    def __init__(self):
        self.input_params_definition = ParamsDefinitionSet()

    def get_input_params_definition(self):
        return self.input_params_definition.get_params_definition_json_list()

    def do_algo(self, input):
        control_params = input.algo_control.control_params
        if not self.check_input_params(self.get_input_params_definition(), control_params):
            log.error("Check input params type error.")
            return None
        mode = input.algo_control.mode
        data = input.algo_data.data
        label = input.algo_data.label
        if mode == 'training':
            try:
                model = linear_model.BayesianRidge(
                    n_iter=control_params["n_iter"],
                    tol=control_params["tol"]
                )
                model.fit(X=data, y=label)
                algo_output = alc.AlgoParam(algo_control={'mode': 'training', 'control_params': ''},
                                                algo_data={'data': data, 'label': label},
                                                algo_model={'model_params': model.get_params(), 'model_instance': model})
            except Exception as e:
                log.error(str(e))
                algo_output = None
        else:
            algo_output = None
        return algo_output
