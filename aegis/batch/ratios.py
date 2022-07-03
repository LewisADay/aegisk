
import botorch
import numpy as np
import torch
from .acquisitions import AcqBaseBatchBO

class EI(AcqBaseBatchBO):
    def __init__(
        self,
        model,
        lb,
        ub,
        under_evaluation,
        n_opt_samples,
        n_opt_bfgs,
    ):

        acq_name = "EI"

        super().__init__(
            model,
            lb,
            ub,
            under_evaluation,
            acq_name,
            n_opt_samples,
            n_opt_bfgs,
        )

    @property
    def acq(self):
        return botorch.acquisition.ExpectedImprovement(self.model, best_f=torch.min(self.model.train_targets), maximize=False)

class UCB(AcqBaseBatchBO):
    def __init__(
        self,
        model,
        lb,
        ub,
        under_evaluation,
        n_opt_samples,
        n_opt_bfgs,
        beta = 0.2,
    ):

        acq_name = "UCB"

        super().__init__(
            model,
            lb,
            ub,
            under_evaluation,
            acq_name,
            n_opt_samples,
            n_opt_bfgs,
        )

        self.beta = beta

    @property
    def acq(self):
        return botorch.acquisition.UpperConfidenceBound(self.model, beta = self.beta, maximize=False)

class EITimeRatio(AcqBaseBatchBO):
    def __init__(
        self,
        model,
        time_model,
        lb,
        ub,
        under_evaluation,
        n_opt_samples,
        n_opt_bfgs,
    ):

        acq_name = "EITimeRatio"

        super().__init__(
            model,
            lb,
            ub,
            under_evaluation,
            acq_name,
            n_opt_samples,
            n_opt_bfgs,
        )

        self.time_model = time_model

    @property
    def acq(self):
        _ei = botorch.acquisition.ExpectedImprovement(self.model, best_f = torch.min(self.model.train_targets), maximize=False)
        _time = self.time_model
        _times = lambda test_x: _time(test_x).means
        return lambda test_x: _ei(test_x) / _times(test_x)

    def update(self, model, under_evaluation, time_model):
        super.update(model, under_evaluation)
        self.time_model = time_model

class UCBTimeRatio():
    pass

class FuncTimeRatio():
    pass