from sklearn import svm
import inanalysis_algo.algo_component as alc
import logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class ParamsDefinitionSet(alc.ParamsDefinitionSet):
    def __init__(self):
        self.params_definition_set =\
            {
                alc.ParamsDefinition(name='gamma', type='float', range='0,1', default_value='auto', description=''),
                alc.ParamsDefinition(name='nu', type='float', range='0,1', default_value='0.5', description=''),
                alc.ParamsDefinition(name='kernel', type='enum', range='linear,poly,rbf,sigmoid,precomputed', default_value='rbf', description=''),
                alc.ParamsDefinition(name='degree', type='int', range='', default_value='3', description=''),
            }


class OneClassSVM(alc.InanalysisAlgo):
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
        if mode == 'training':
            try:
                model = svm.OneClassSVM(
                    nu=control_params["nu"],
                    kernel=control_params["kernel"],
                    gamma=control_params["gamma"],
                    degree=control_params["degree"]
                )
                model.fit(data)
                algo_output = alc.AlgoParam(algo_control={'mode': 'training', 'control_params': ''},
                                            algo_data={'data': data, 'label': None},
                                            algo_model={'model_params': model.get_params(), 'model_instance': model})
            except Exception as e:
                log.error(str(e))
                algo_output = None
        else:
            algo_output = None
        return algo_output
