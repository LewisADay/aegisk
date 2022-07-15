
from msilib.schema import Error
import botorch
from .batch.ratios import CostAcqFunc, EICostAcq, UCBCostAcq


"""
class SelectiveKillingAcqBase(botorch.acquisition.AnalyticAcquisitionFunction):
    def __init__(
        self,
        model,
        cost_model,
        T_data,
        T_cost,
        under_eval,
        posterior_transform = None,
        maximize: bool = True,
        **kwargs,
    ) -> None:

        super().__init__(
            model=model,
            posterior_transform=posterior_transform,
            **kwargs
            )

        self.model = model
        self.cost_model = cost_model
        self.T_data = T_data
        self.T_cost = T_cost
        self.under_eval = under_eval
        self.maximize = maximize

class ScalarKillingAcq(SelectiveKillingAcqBase):
    def __init__(
        self,
        model,
        cost_model,
        T_data,
        T_cost,
        posterior_transform = None,
        maximize: bool = True,
        **kwargs,
        ):

        super().__init__(
            model,
            cost_model,
            T_data,
            T_cost,
            posterior_transform = posterior_transform,
            maximize = maximize,
            **kwargs,
        )

    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: Tensor) -> Tensor:
        # Get the value of the point(s) X
        

"""

class SelectiveKillingBase(CostAcqFunc):
    def __init__(
        self,
        model,
        cost_model,
        T_data,
        T_cost,
        lb,
        ub,
        under_evaluation,
        kill_name,
        n_opt_samples,
        n_opt_bfgs,
    ):

        super().__init__(
            model,
            cost_model,
            T_data,
            T_cost,
            lb,
            ub,
            under_evaluation,
            kill_name,
            n_opt_samples,
            n_opt_bfgs,
        )

    def _opt_acq(self):

        # perform Boltzmann sampling with n_opt_samples samples and
        # optimise the best n_opt_bfgs of these with l-bfgs-b

        n_samples = self.n_opt_samples / 2

        # try a few times just in case we get unlucky and all our samples
        # are in flat regions of space (unlikely but can happen with EI)
        MAX_ATTEMPTS = 5

        e = RuntimeError
        for attempts in range(MAX_ATTEMPTS):
            n_samples *= 2

            try:
                train_xnew, acq_f = botorch.optim.optimize_acqf(
                    acq_function=self.acq,
                    q=1,
                    bounds=self.problem_bounds,
                    num_restarts=self.n_opt_bfgs,
                    raw_samples=self.n_opt_samples,
                )
                return train_xnew

            # botorch throws a RuntimeError with the reason:
            # invalid multinomial distribution (sum of probabilities <= 0)
            except RuntimeError:
                e = RuntimeError.with_traceback
                continue

        # if we've reached this point we've failed to get a valid location,
        # so raise an exception
        msg = "Failed to optimise the acquisition function after"
        msg += f" {MAX_ATTEMPTS} attempts. The acquisition function had a "
        msg += " sum of probabilities <= 0 every time. This is very unlikely!"
        raise e

    def _get_next(self, x_star):
        raise NotImplementedError

    def possible_killing(self):
        raise NotImplementedError
        
class ScalarKilling(SelectiveKillingBase):
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
        delta,
    ):

        kill_name = "ScalarKilling"

        super().__init__(
            model,
            cost_model,
            T_data,
            T_cost,
            lb,
            ub,
            under_evaluation,
            kill_name,
            n_opt_samples,
            n_opt_bfgs,
        )

        self.delta = delta

    @property
    def acq(self):
        return EICostAcq(
            model = self.model,
            cost_model = self.cost_model,
            T_data = self.T_data,
            T_cost = self.T_cost,
            maximize = self.maximize,
        )

    def value(self, x):
        return self.acq(x)

    def _get_next(self, x_star):
        print("-------")
        print("UE:")
        print(self.ue)
        print("-------")

    def possible_killing(self):
        return True
