from sklearn import linear_model
import inanalysis_algo.algo_component as alc
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class ParamsDefinitionSet(alc.ParamsDefinitionSet):
    def __init__(self):
        self.params_definition_set =\
            {
                alc.ParamsDefinition(name='fit_intercept', type='boolean', range='True,False', default_value='True', description=''),
                alc.ParamsDefinition(name='normalize', type='boolean', range='True,False', default_value='False', description=''),
                alc.ParamsDefinition(name='copy_X', type='boolean', range='True,False', default_value='True', description=''),
                alc.ParamsDefinition(name='n_jobs', type='int', range='', default_value='1', description=''),
                alc.ParamsDefinition(name='degree', type='int', range='', default_value='1', description='')
            }


class PolynomialRegression(alc.InanalysisAlgo):
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
                regr = linear_model.LinearRegression(
                    fit_intercept=control_params["fit_intercept"],
                    normalize=control_params["normalize"],
                    copy_X=control_params["copy_X"],
                    n_jobs=control_params["n_jobs"]
                )
                degree = control_params["degree"]
                model = make_pipeline(PolynomialFeatures(degree), regr)
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
