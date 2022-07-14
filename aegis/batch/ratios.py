
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

class EICostAcq(botorch.acquisition.AnalyticAcquisitionFunction):
    def __init__(
        self,
        model,
        cost_model,
        T_data,
        T_cost,
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
        self.cost_model = cost_model
        self.T_data = T_data
        self.T_cost = T_cost
        self.best_f = best_f

    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: Tensor) -> Tensor:
        ei = botorch.acquisition.ExpectedImprovement(
            self.model,
            best_f = self.best_f,
            maximize = self.maximize
        )

        ei = ei.forward(X)

        et = self.cost_model(X)
        et = self.T_cost.unscale_mean(et.mean.ravel())

        return torch.div(ei, et)

class UCBCostAcq(botorch.acquisition.AnalyticAcquisitionFunction):
    def __init__(
        self,
        model,
        cost_model,
        T_data,
        T_cost,
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
        self.cost_model = cost_model
        self.T_data = T_data
        self.T_cost = T_cost
        self.beta = beta

    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: Tensor) -> Tensor:
        ucb = botorch.acquisition.UpperConfidenceBound(
            self.model,
            beta= self.beta,
            maximize = self.maximize
        )

        ucb = ucb.forward(X)

        et = self.cost_model(X)
        et = self.T_cost.unscale_mean(et.mean.ravel())

        return torch.div(ucb, et)


class FuncCostRatio(botorch.acquisition.AnalyticAcquisitionFunction):
    def __init__(
        self,
        model,
        cost_model,
        T_data,
        T_cost,
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
        self.cost_model = cost_model
        self.T_data = T_data
        self.T_cost = T_cost


    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: Tensor) -> Tensor:
        ev = self.model(X)
        ev = self.T_data.unscale_mean(ev.mean.ravel())

        et = self.cost_model(X)
        et = self.T_cost.unscale_mean(et.mean.ravel())

        return torch.div(-ev, et)

class CostAcqFunc(AcqBaseBatchBO):
    def __init__(
        self,
        model,
        cost_model,
        T_data,
        T_cost,
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

        self.cost_model = cost_model
        self.T_data = T_data
        self.T_cost = T_cost

    def update(self, model, under_evaluation, cost_model):
        """
        Updates the acquisition function with the latest model and locations
        under evaluation.
        """
        self.model = model
        self.ue = under_evaluation
        self.cost_model = cost_model
        self.updated = True


class EICostRatio(CostAcqFunc):
    def __init__(
        self,
        model,
        cost_model,
        T_data,
        T_cost,
        lb,
        ub,
        under_evaluation,
        n_opt_samples,
        n_opt_bfgs,
    ):

        acq_name = "EICostRatio"

        super().__init__(
            model,
            cost_model,
            T_data,
            T_cost,
            lb,
            ub,
            under_evaluation,
            acq_name,
            n_opt_samples,
            n_opt_bfgs,
        )

    @property
    def acq(self):
        return EICostAcq(
            model = self.model,
            cost_model = self.cost_model,
            T_data = self.T_data,
            T_cost = self.T_cost,
            best_f = self.model.train_targets.min(),
        )

class UCBCostRatio(CostAcqFunc):
    def __init__(
        self,
        model,
        cost_model,
        T_data,
        T_cost,
        lb,
        ub,
        under_evaluation,
        n_opt_samples,
        n_opt_bfgs,
    ):

        acq_name = "UCBCostRatio"

        super().__init__(
            model,
            cost_model,
            T_data,
            T_cost,
            lb,
            ub,
            under_evaluation,
            acq_name,
            n_opt_samples,
            n_opt_bfgs,
        )

    @property
    def acq(self):
        return UCBCostAcq(
            model = self.model,
            cost_model = self.cost_model,
            T_data = self.T_data,
            T_cost = self.T_cost,
            beta = 2,
        )

class FuncCostRatio(CostAcqFunc):
    def __init__(
        self,
        model,
        cost_model,
        T_data,
        T_cost,
        lb,
        ub,
        under_evaluation,
        n_opt_samples,
        n_opt_bfgs,
    ):

        acq_name = "FuncCostRatio"

        super().__init__(
            model,
            cost_model,
            T_data,
            T_cost,
            lb,
            ub,
            under_evaluation,
            acq_name,
            n_opt_samples,
            n_opt_bfgs,
        )

    @property
    def acq(self):
        return FuncCostRatio(
            model = self.model,
            cost_model = self.cost_model,
            T_data = self.T_data,
            T_cost = self.T_cost,
        )