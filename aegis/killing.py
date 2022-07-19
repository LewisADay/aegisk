
import botorch
import torch
from aegis import batch
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
        eval_times,
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

        self.eval_times = eval_times

    def update(self, model, under_evaluation, cost_model, eval_times):
        super().update(model, under_evaluation, cost_model)
        self.eval_times = eval_times

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

class NoKilling(SelectiveKillingBase):
    def __init__(
        self,
        model,
        cost_model,
        T_data,
        T_cost,
        lb,
        ub,
        under_evaluation,
        eval_times,
        n_opt_samples,
        n_opt_bfgs,
    ):

        kill_name = "NoKilling"

        super().__init__(
            model,
            cost_model,
            T_data,
            T_cost,
            lb,
            ub,
            under_evaluation,
            eval_times,
            kill_name,
            n_opt_samples,
            n_opt_bfgs,
        )
    
    def _get_next(self, x_star):
        return None

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
        eval_times,
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
            eval_times,
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
            best_f = self.model.train_targets.min()
        )

    def _get_index_of_x(self, x):
        for k in range(len(self.ue)):
            if x == self.ue[k]:
                return k

    def _ongoing_value(self, x):
        ei = botorch.acquisition.ExpectedImprovement(
            self.model,
            best_f = self.model.train_targets.min(),
            maximize=False
        )

        ei = ei.forward(x)

        ec = self.cost_model(x)
        ec = self.T_cost.unscale_mean(ec.mean.ravel())
        ec = ec - self.eval_times[self._get_index_of_x(x)]
        if ec < 0:
            ec = 0

        return torch.div(ei, ec)

    def value(self, x):
        if x in self.ue:
            return self._ongoing_value(x)
        else:
            return self.acq(x)

    def eligibility_criteria(self, x_star, x_i):
        print("%%%%%%%%%%%%%%%%%%%")
        print(f"x_star: {x_star[0][0]}, val:{self.value(x_star)[0]}")
        print(f"x_i: {x_i[0][0]}, val:{self.value(x_i)[0]}")
        print("%%%%%%%%%%%%%%%%%%%")
        if self.value(x_star) > self.value(x_i) + self.delta:
            return True
        else:
            return False

    def decision(self, x_is):
        for x_i in x_is:
            adopt = True
            print(f"here: {x_i}") #################
            for x_j in [_ for _ in x_is if _ != x_i]:
                if self.value(x_j) < self.value(x_i):
                    adopt = False
            if adopt:
                print(f"Returning: {x_i}")
                return x_i

        # If none met our criteria
        return None

    def _get_next(self, x_star):
        x_is = []
        for x_i in self.ue:
            x_i = torch.reshape(x_i, (1,len(x_i)))
            if self.eligibility_criteria(x_star, x_i):
                x_is.append(x_i)
        
        # If we have no eligible evaluations return
        if x_is == []:
            return None
        
        # Otherwise we must decide between them
        #x_is = torch.as_tensor(x_is)
        x_i = self.decision(x_is)

        return x_i

class DeterministicKilling(SelectiveKillingBase):
    def __init__(
        self,
        model,
        cost_model,
        T_data,
        T_cost,
        lb,
        ub,
        under_evaluation,
        eval_times,
        n_opt_samples,
        n_opt_bfgs,
        delta,
        acq_name,
        acq_params,
    ):

        kill_name = "DeterministicKilling"

        super().__init__(
            model,
            cost_model,
            T_data,
            T_cost,
            lb,
            ub,
            under_evaluation,
            eval_times,
            kill_name,
            n_opt_samples,
            n_opt_bfgs,
        )

        self.delta = delta
        self._acq_class = getattr(batch, acq_name)
        self._acq_params = acq_params

    def _acq(self, ue):
        return self._acq_class(
                model=self.model,
                cost_model=self.cost_model,
                T_data=self.T_data,
                T_cost=self.T_cost,
                lb=self.lb,
                ub=self.ub,
                under_evaluation=ue,
                **self._acq_params,
            )

    @property
    def acq(self):
        return self._acq_class(
                model=self.model,
                cost_model=self.cost_model,
                T_data=self.T_data,
                T_cost=self.T_cost,
                lb=self.lb,
                ub=self.ub,
                under_evaluation=self.ue,
                **self._acq_params,
            )


    def _get_index_of_x(self, x):
        for k in range(len(self.ue)):
            if x == self.ue[k]:
                return k

    def _ongoing_value(self, x):

        tmp_ue = torch.as_tensor([_x for _x in self.ue if _x != x])
        tmp_acq = self._acq(tmp_ue)

        numerator = tmp_acq.acq.forward(x)

        ec = self.cost_model(x)
        ec = self.T_cost.unscale_mean(ec.mean.ravel())
        ec = ec - self.eval_times[self._get_index_of_x(x)]
        if ec < 0:
            ec = 0

        return torch.div(numerator, ec)

    def _candidate_value(self, x):

        self.acq.update(self.model, self.ue, self.cost_model)

        numerator = self.acq.acq.forward(x)

        ec = self.cost_model(x)
        ec = self.T_cost.unscale_mean(ec.mean.ravel())
        if ec < 0:
            ec = 0

        return torch.div(numerator, ec)

    def value(self, x):
        if x in self.ue:
            return self._ongoing_value(x)
        else:
            return self._candidate_value(x)

    def eligibility_criteria(self, x_star, x_i):
        print("%%%%%%%%%%%%%%%%%%%")
        print(f"x_star: {x_star[0][0]}, val:{self.value(x_star)[0]}")
        print(f"x_i: {x_i[0][0]}, val:{self.value(x_i)[0]}")
        print("%%%%%%%%%%%%%%%%%%%")
        if self.value(x_star) > self.value(x_i) + self.delta:
            return True
        else:
            return False

    def decision(self, x_is):
        for x_i in x_is:
            adopt = True
            print(f"here: {x_i}") #################
            for x_j in [_ for _ in x_is if _ != x_i]:
                if self.value(x_j) < self.value(x_i):
                    adopt = False
            if adopt:
                print(f"Returning: {x_i}")
                return x_i

        # If none met our criteria
        return None

    def _get_next(self, x_star):
        x_is = []
        for x_i in self.ue:
            x_i = torch.reshape(x_i, (1,len(x_i)))
            if self.eligibility_criteria(x_star, x_i):
                x_is.append(x_i)
        
        # If we have no eligible evaluations return
        if x_is == []:
            return None
        
        # Otherwise we must decide between them
        #x_is = torch.as_tensor(x_is)
        x_i = self.decision(x_is)

        return x_i

class ProabilisticKilling(SelectiveKillingBase):
    def __init__(
        self,
        model,
        cost_model,
        T_data,
        T_cost,
        lb,
        ub,
        under_evaluation,
        eval_times,
        n_opt_samples,
        n_opt_bfgs,
        p,
        acq_name,
        acq_params,
    ):

        kill_name = "ProabilisticKilling"

        super().__init__(
            model,
            cost_model,
            T_data,
            T_cost,
            lb,
            ub,
            under_evaluation,
            eval_times,
            kill_name,
            n_opt_samples,
            n_opt_bfgs,
        )

        self.p = p
        self._acq_class = getattr(batch, acq_name)
        self._acq_params = acq_params

    def _acq(self, ue):
        return self._acq_class(
                model=self.model,
                cost_model=self.cost_model,
                T_data=self.T_data,
                T_cost=self.T_cost,
                lb=self.lb,
                ub=self.ub,
                under_evaluation=ue,
                **self._acq_params,
            )

    @property
    def acq(self):
        return self._acq_class(
                model=self.model,
                cost_model=self.cost_model,
                T_data=self.T_data,
                T_cost=self.T_cost,
                lb=self.lb,
                ub=self.ub,
                under_evaluation=self.ue,
                **self._acq_params,
            )


    def _get_index_of_x(self, x):
        for k in range(len(self.ue)):
            if x == self.ue[k]:
                return k

    def _ongoing_value(self, x):

        tmp_ue = torch.as_tensor([_x for _x in self.ue if _x != x])
        tmp_acq = self._acq(tmp_ue)

        numerator = tmp_acq.acq.forward(x)

        ec = self.cost_model(x)
        ec = self.T_cost.unscale_mean(ec.mean.ravel())
        ec = ec - self.eval_times[self._get_index_of_x(x)]
        if ec < 0:
            ec = 0

        return torch.div(numerator, ec)

    def _candidate_value(self, x):

        self.acq.update(self.model, self.ue, self.cost_model)

        numerator = self.acq.acq.forward(x)

        ec = self.cost_model(x)
        ec = self.T_cost.unscale_mean(ec.mean.ravel())
        if ec < 0:
            ec = 0

        return torch.div(numerator, ec)

    def value(self, x):
        if x in self.ue:
            return self._ongoing_value(x)
        else:
            return self._candidate_value(x)

    def eligibility_criteria(self, x_star, x_i):
        print("%%%%%%%%%%%%%%%%%%%")
        print(f"x_star: {x_star[0][0]}, val:{self.value(x_star)[0]}")
        print(f"x_i: {x_i[0][0]}, val:{self.value(x_i)[0]}")
        print("%%%%%%%%%%%%%%%%%%%")
        if self.value(x_star) > self.value(x_i) + self.delta:
            return True
        else:
            return False

    def decision(self, x_is):
        for x_i in x_is:
            adopt = True
            print(f"here: {x_i}") #################
            for x_j in [_ for _ in x_is if _ != x_i]:
                if self.value(x_j) < self.value(x_i):
                    adopt = False
            if adopt:
                print(f"Returning: {x_i}")
                return x_i

        # If none met our criteria
        return None

    def _get_next(self, x_star):
        x_is = []
        for x_i in self.ue:
            x_i = torch.reshape(x_i, (1,len(x_i)))
            if self.eligibility_criteria(x_star, x_i):
                x_is.append(x_i)
        
        # If we have no eligible evaluations return
        if x_is == []:
            return None
        
        # Otherwise we must decide between them
        #x_is = torch.as_tensor(x_is)
        x_i = self.decision(x_is)

        return x_i
