from learn_tools.uncertain_learner import UncertainLearner
from learn_tools.active_nn import rf_predict, RF_CLASSIFIER, RF_REGRESSOR, get_sklearn_model, retrain_sklearer
from learn_tools.active_learner import THRESHOLD

import numpy as np

BATCH_RF = (RF_REGRESSOR, RF_CLASSIFIER)

BATCH_MAXVAR_RF = 'rf_batch_maxvar'
BATCH_STRADDLE_RF = 'rf_batch_straddle'
BATCH_ACTIVE_RF = (BATCH_MAXVAR_RF, BATCH_STRADDLE_RF)

RF_MODELS = BATCH_RF + BATCH_ACTIVE_RF

class ActiveRF(UncertainLearner):

    USE_GRADIENTS = False

    def __init__(self, func, initx=np.array([[]]), inity=np.array([]),
                 model_type=RF_REGRESSOR, use_sample_weights=False, **kwargs):
        print('{} using {}'.format(self.__class__.__name__, model_type))
        if model_type == RF_CLASSIFIER:
            inity = (THRESHOLD < inity).astype(float)
        super(ActiveRF, self).__init__(func, initx=initx, inity=inity, **kwargs)
        self.model_type = model_type
        assert self.model_type in RF_MODELS
        self.model = get_sklearn_model(self.model_type)
        self.use_sample_weights = use_sample_weights
        self.name = '{}-{}'.format(self.model_type, self.query_type).lower()

    def predict(self, X, **kwargs):
        mu, var = rf_predict(self.model, X)
        mu = mu.reshape(-1, 1)
        var = var.reshape(-1, 1)
        # TODO: sample_weight
        #print(self.func.inputs)
        #print(self.lengthscale.round(3).tolist())
        return mu, var

    def predictive_gradients(self, x, **kwargs):
        raise NotImplementedError()

    @property
    def lengthscale(self):
        # Higher feature importance => more important
        return np.reciprocal(self.model.feature_importances_)

    def metric(self):
        pass

    def retrain(self, **kwargs):
        retrain_sklearer(self, **kwargs)
        print('Num Train: {} | Train R^2: {:.3f} | OOB R^2: {:.3f}'.format(
            len(self.xx), self.model.score(self.xx, self.yy), self.model.oob_score_))
        print('Significant (most to least): {{{}}}'.format(', '.join( # TODO: could split into context vs parameter
            '{}: {:.3f}'.format(n, s) for s, n in sorted(zip(self.lengthscale, self.func.inputs)))))
