from sklearn import tree
import inanalysis_algo.algo_component as alc
import logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class ParamsDefinitionSet(alc.ParamsDefinitionSet):
    def __init__(self):
        self.params_definition_set =\
            {
                alc.ParamsDefinition(name='max_depth', type='int', range='', default_value='None', description='')
            }


class DCtree(alc.InanalysisAlgo):
    def __init__(self):
        self.input_params_definition = ParamsDefinitionSet()

    def get_input_params_definition(self):
        return self.input_params_definition.get_params_definition_json_list()

    def do_algo(self, input_params):
        control_params = input_params.algo_control.control_params
        if not self.check_input_params(self.get_input_params_definition(), control_params):
            log.error("Check input params type error.")
            return None
        data = input_params.algo_data.data
        label = input_params.algo_data.label
        mode = input_params.algo_control.mode
        if mode == 'training':
            try:
                model = tree.DecisionTreeClassifier(
                    max_depth=control_params["max_depth"]
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
