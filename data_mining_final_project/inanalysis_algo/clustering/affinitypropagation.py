from sklearn import cluster
import inanalysis_algo.algo_component as alc
from sklearn import metrics
import logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class ParamsDefinitionSet(alc.ParamsDefinitionSet):
    def __init__(self):
        self.params_definition_set =\
            {
                alc.ParamsDefinition(name='damping', type='float', range='0.5,1', default_value='0.5',
                                     description='0.5 >= value > 1. Control the convergence rate of algorithm.'),
                alc.ParamsDefinition(name='preference', type='float', range='', default_value='-50',
                                     description='Control the number of clusters, larger makes more clusters.'),
                alc.ParamsDefinition(name='convergence_iter', type='int', range='', default_value='15', description=''),
                alc.ParamsDefinition(name='max_iter', type='int', range='', default_value='200', description='')
            }


class AffinityPropagation(alc.InanalysisAlgo):
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
                model = cluster.AffinityPropagation(
                    damping=control_params["damping"],
                    preference=control_params["preference"],
                    convergence_iter=control_params["convergence_iter"],
                    max_iter=control_params["max_iter"]
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
