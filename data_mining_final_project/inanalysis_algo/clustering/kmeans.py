from sklearn import cluster
import inanalysis_algo.algo_component as alc
import logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class ParamsDefinitionSet(alc.ParamsDefinitionSet):
    def __init__(self):
        self.params_definition_set =\
            {
                alc.ParamsDefinition(name='n_clusters', type='int', range='', default_value='8', description=''),
                alc.ParamsDefinition(name='n_init', type='int', range='', default_value='10', description=''),
                alc.ParamsDefinition(name='max_iter', type='int', range='', default_value='300', description='')
            }


class Kmeans(alc.InanalysisAlgo):
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
                model = cluster.KMeans(
                    n_clusters=control_params["n_clusters"],
                    n_init=control_params["n_init"],
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







