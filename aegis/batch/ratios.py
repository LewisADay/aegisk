
import botorch
import numpy as np
import torch
from torch import Tensor
from torch.distributions import Normal
from .acquisitions import AcqBaseBatchBO
from botorch.utils.transforms import t_batch_mode_transform
import botorch

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
        return botorch.acquisition.ExpectedImprovement(self.model, best_f=self.model.train_targets.min(), maximize=False)

class UCB(AcqBaseBatchBO):
    def __init__(
        self,
        model,
        lb,
        ub,
        under_evaluation,
        n_opt_samples,
        n_opt_bfgs,
        beta = 2,
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

class EITimeAcq(botorch.acquisition.AnalyticAcquisitionFunction):
    def __init__(
        self,
        model,
        time_model,
        T_data,
        T_time,
        best_f,
        posterior_transform = None,
        maximize: bool = False,
        **kwargs,
    ) -> None:

        super().__init__(
            model=model,
            posterior_transform=posterior_transform,
            **kwargs
            )

        self.maximize = maximize
        self.time_model = time_model
        self.T_data = T_data
        self.T_time = T_time
        self.best_f = best_f

    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: Tensor) -> Tensor:
        ei = botorch.acquisition.ExpectedImprovement(
            self.model,
            best_f = self.best_f,
            maximize = self.maximize
        )

        ei = ei.forward(X)

        et = self.time_model(X)
        et = self.T_time.unscale_mean(et.mean.ravel())

        return torch.div(ei, et)

class UCBTimeAcq(botorch.acquisition.AnalyticAcquisitionFunction):
    def __init__(
        self,
        model,
        time_model,
        T_data,
        T_time,
        beta,
        posterior_transform = None,
        maximize: bool = False,
        **kwargs,
    ) -> None:

        super().__init__(
            model=model,
            posterior_transform=posterior_transform,
            **kwargs
            )

        self.maximize = maximize
        self.time_model = time_model
        self.T_data = T_data
        self.T_time = T_time
        self.beta = beta

    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: Tensor) -> Tensor:
        ucb = botorch.acquisition.UpperConfidenceBound(
            self.model,
            beta= self.beta,
            maximize = self.maximize
        )

        ucb = ucb.forward(X)

        et = self.time_model(X)
        et = self.T_time.unscale_mean(et.mean.ravel())

        return torch.div(ucb, et)


class FuncTimeRatio():
    def __init__(
        self,
        model,
        time_model,
        T_data,
        T_time,
        posterior_transform = None,
        maximize: bool = False,
        **kwargs,
    ) -> None:

        super().__init__(
            model=model,
            posterior_transform=posterior_transform,
            **kwargs
            )

        self.maximize = maximize
        self.time_model = time_model
        self.T_data = T_data
        self.T_time = T_time


    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: Tensor) -> Tensor:
        ev = self.model(X)
        ev = self.T_data.unscale_mean(ev.mean.ravel())

        et = self.time_model(X)
        et = self.T_time.unscale_mean(et.mean.ravel())

        return torch.div(-ev, et)

class TimeAcqFunc(AcqBaseBatchBO):
    def __init__(
        self,
        model,
        time_model,
        T_data,
        T_time,
        lb,
        ub,
        under_evaluation,
        acq_name,
        n_opt_samples,
        n_opt_bfgs,
    ):

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
        self.T_data = T_data
        self.T_time = T_time

    def update(self, model, under_evaluation, time_model):
        """
        Updates the acquisition function with the latest model and locations
        under evaluation.
        """
        self.model = model
        self.ue = under_evaluation
        self.time_model = time_model
        self.updated = True


class EITimeRatio(TimeAcqFunc):
    def __init__(
        self,
        model,
        time_model,
        T_data,
        T_time,
        lb,
        ub,
        under_evaluation,
        n_opt_samples,
        n_opt_bfgs,
    ):

        acq_name = "EITimeRatio"

        super().__init__(
            model,
            time_model,
            T_data,
            T_time,
            lb,
            ub,
            under_evaluation,
            acq_name,
            n_opt_samples,
            n_opt_bfgs,
        )

    @property
    def acq(self):
        return EITimeAcq(
            model = self.model,
            time_model=self.time_model,
            T_data = self.T_data,
            T_time = self.T_time,
            best_f = self.model.train_targets.min(),
        )

class UCBTimeRatio(TimeAcqFunc):
    def __init__(
        self,
        model,
        time_model,
        T_data,
        T_time,
        lb,
        ub,
        under_evaluation,
        n_opt_samples,
        n_opt_bfgs,
    ):

        acq_name = "UCBTimeRatio"

        super().__init__(
            model,
            time_model,
            T_data,
            T_time,
            lb,
            ub,
            under_evaluation,
            acq_name,
            n_opt_samples,
            n_opt_bfgs,
        )

    @property
    def acq(self):
        return UCBTimeAcq(
            model = self.model,
            time_model=self.time_model,
            T_data = self.T_data,
            T_time = self.T_time,
            beta = 2,
        )

class FuncTimeRatio(TimeAcqFunc):
    def __init__(
        self,
        model,
        time_model,
        T_data,
        T_time,
        lb,
        ub,
        under_evaluation,
        n_opt_samples,
        n_opt_bfgs,
    ):

        acq_name = "FuncTimeRatio"

        super().__init__(
            model,
            time_model,
            T_data,
            T_time,
            lb,
            ub,
            under_evaluation,
            acq_name,
            n_opt_samples,
            n_opt_bfgs,
        )

    @property
    def acq(self):
        return FuncTimeRatio(
            model = self.model,
            time_model=self.time_model,
            T_data = self.T_data,
            T_time = self.T_time,
        )