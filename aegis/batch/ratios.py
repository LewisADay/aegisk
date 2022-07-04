
import botorch
import numpy as np
import torch
from torch import Tensor
from torch.distributions import Normal
from .acquisitions import AcqBaseBatchBO
from botorch.utils.transforms import t_batch_mode_transform

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

class EITimeAcq(botorch.acquisition.AnalyticAcquisitionFunction):
    def __init__(
        self,
        model,
        time_model,
        best_f,
        posterior_transform = None,
        maximize: bool = True,
        **kwargs,
    ) -> None:

        super().__init__(
            model=model,
            posterior_transform=posterior_transform,
            **kwargs
            )

        self.maximize = maximize
        self.time_model = time_model

        if not torch.is_tensor(best_f):
            best_f = torch.tensor(best_f)

        self.register_buffer("best_f", best_f)

    def _ei(self, X: Tensor) -> Tensor:
        self.best_f = self.best_f.to(X)
        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        mean = posterior.mean
        # deal with batch evaluation and broadcasting
        view_shape = mean.shape[:-2] if mean.shape[-2] == 1 else mean.shape[:-1]
        mean = mean.view(view_shape)
        sigma = posterior.variance.clamp_min(1e-9).sqrt().view(view_shape)
        u = (mean - self.best_f.expand_as(mean)) / sigma
        if not self.maximize:
            u = -u
        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))
        ei = sigma * (updf + u * ucdf)
        return ei

    def forward(self, X: Tensor) -> Tensor:
        ei = self._ei(X)
        return ei / self.time_model(X).mean

    def __call__(self, X: Tensor) -> Tensor:
        return self.forward(X)


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
        return EITimeAcq(
            model = self.model,
            time_model=self.time_model,
            best_f = self.model.train_targets.min(),
        )

    def update(self, model, under_evaluation, time_model):
        """
        Updates the acquisition function with the latest model and locations
        under evaluation.
        """
        self.model = model
        self.ue = under_evaluation
        self.time_model = time_model
        self.updated = True

class UCBTimeRatio():
    pass

class FuncTimeRatio():
    pass