from sklearn import ensemble
import inanalysis_algo.algo_component as alc
import logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class ParamsDefinitionSet(alc.ParamsDefinitionSet):
    def __init__(self):
        self.params_definition_set =\
            {
                alc.ParamsDefinition(name='n_estimators', type='int', range='', default_value='100', description='number of base estimators in the ensemble'),
                alc.ParamsDefinition(name='contamination', type='float', range='0.0,0.5', default_value='0.1', description='proportion of outliers in the data set'),
                alc.ParamsDefinition(name='n_jobs', type='int', range='', default_value='1', description='number of jobs to run in parallel for both fit and predict')
            }


class IsolationForest(alc.InanalysisAlgo):
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
                model = ensemble.IsolationForest(
                    n_estimators=control_params["n_estimators"],
                    contamination=control_params["contamination"],
                    n_jobs=control_params["n_jobs"]
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
