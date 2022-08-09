
import botorch
import torch
import numpy as np
from aegis import batch, util
from .batch.ratios import CostAcqFunc, EICostAcq, UCBCostAcq
import matplotlib.pyplot as plt
from scipy.stats import norm
import plotting

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
        return self._acq(self.ue)


    def __get_index_of_x(self, x):
        for k in range(len(self.ue)):
            if torch.equal(x, self.ue[k]):
                print(k)
                return k

    def _get_index_of_x(self, x):
        return torch.eq(self.ue, x).nonzero()[0,0]

    def _ongoing_value(self, x):

        tensor_list = [torch.as_tensor(_x) for _x in self.ue if not torch.equal(_x, x)]
        if len(tensor_list) != 0:
            tmp_ue = torch.stack(tensor_list)
        else:
            tmp_ue = torch.tensor([])
        tmp_acq = self._acq(tmp_ue)

        _x = torch.reshape(x, (1,len(x)))
        numerator = tmp_acq.acq.forward(_x)

        ec = self.cost_model(_x)
        ec = self.T_cost.unscale_mean(ec.mean.ravel())
        ec = ec - self.eval_times[self._get_index_of_x(x)]
        if ec < 0:
            ec = 0

        return torch.div(numerator, ec)

    def _candidate_value(self, x):

        numerator = self.acq.acq.forward(x)

        ec = self.cost_model(x)
        ec = self.T_cost.unscale_mean(ec.mean.ravel())
        if ec < 0:
            ec = 0

        return torch.div(numerator, ec)

    def value(self, x):
        if any([torch.equal(x, ongoing) for ongoing in self.ue]):
            return self._ongoing_value(x)
        else:
            return self._candidate_value(x)

    def eligibility_criteria(self, x_star, x_i):
        if self.value(x_star) / self.value(x_i) > self.delta:
            return True
        else:
            return False

    def decision(self, x_is):
        for x_i in x_is:
            adopt = True
            for x_j in [_ for _ in x_is if not torch.equal(_, x_i)]:
                if self.value(x_j) < self.value(x_i):
                    adopt = False
            if adopt:
                return x_i

        # If none met our criteria
        return None

    def _get_next(self, x_star):
        x_is = []
        for x_i in self.ue:
            if self.eligibility_criteria(x_star, x_i):
                x_i = torch.reshape(x_i, (1,len(x_i)))
                x_is.append(x_i)
        
        # If we have no eligible evaluations return
        if x_is == []:
            return None
        
        # Otherwise we must decide between them
        #x_is = torch.as_tensor(x_is)
        x_i = self.decision(x_is)

        return x_i

class ProbabilisticKilling(SelectiveKillingBase):
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
        alpha=0.8,
        n_samples=100,
        epsilon=1e-8
    ):

        kill_name = "ProbabilisticKilling"

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

        self.alpha = alpha
        self.n_samples = n_samples
        self.epsilon = epsilon

    def __get_index_of_x(self, x):
        for k in range(len(self.ue)):
            if x == self.ue[k]:
                return k

    def _get_index_of_x(self, x):
        return torch.eq(self.ue, x).nonzero()[0,0]

    def _ongoing_cost(self, x):
        if x not in self.ue:
            return 0
        else:
            return self.eval_times[self._get_index_of_x(x)]


    def _value_distribution(self, x):

        # Get distribution of problem model at x
        Y_x = self.model(x)

        # Extract standard deviation
        sig_p = Y_x.stddev

        # Scale back to actual output to extract mean
        mu_p = self.T_data.unscale_mean(Y_x.mean.ravel())

        # Generate samples from Y_x
        y_x = np.random.normal(mu_p.detach(), sig_p.detach(), self.n_samples)

        # Convert to samples of the improvement
        f_star = self.model.train_targets.min()
        f_star = self.T_data.unscale_mean(f_star)
        i_x = np.array([f_star - sample for sample in y_x])
        i_x[i_x < 0] = 0


        # Get distribution of cost model at x
        C_x = self.cost_model(x)

        # Extract standard deviation
        sig_c = C_x.stddev

        # Scale back to actual output to extract mean
        mu_c = self.T_cost.unscale_mean(C_x.mean.ravel())

        # Get spent cost
        spent_cost = self._ongoing_cost(x)

        # Generate samples from C_x - spent_cost
        c_x = np.random.normal((mu_c - spent_cost).detach(), sig_c.detach(), self.n_samples)

        # Clamp c_x >= epsilon
        c_x[c_x < self.epsilon] = self.epsilon

        # Calculate value distribution samples v_x
        v_x = [i_x[k] / c_x[k] for k in range(self.n_samples)]

        return v_x

    def _probability(self, candidate_samples, ongoing_samples):

        counter = 0
        for candidate_sample, ongoing_sample in zip(candidate_samples, ongoing_samples):
            if candidate_sample > ongoing_sample:
                counter += 1

        return counter / len(candidate_samples)

    def eligibility_criteria(self, x_i):
        x_i_samples = self._value_distribution(x_i)
        p = self._probability(self.x_star_samples, x_i_samples)
        if p > self.alpha:
            return (True, p)
        else:
            return (False, p)

    def decision(self, x_is):
        for x_i, i_p in x_is:
            adopt = True
            for j_p in [_[1] for _ in x_is if not torch.equal(_[0], x_i)]:
                if j_p > i_p:
                    adopt = False
            if adopt:
                return x_i

        # If none met our criteria
        return None

    def _get_next(self, x_star):
        self.x_star_samples = self._value_distribution(x_star)
        x_is = []
        for x_i in self.ue:
            x_i = torch.reshape(x_i, (1,len(x_i)))
            eligible, p = self.eligibility_criteria(x_i)
            if eligible:
                x_is.append((x_i, p))
        
        # If we have no eligible evaluations return
        if x_is == []:
            return None
        
        # Otherwise we must decide between them
        #x_is = torch.as_tensor(x_is)
        x_i = self.decision(x_is)

        # do some validation plots
        #self._temp_plots(x_star, x_i)

        return x_i

    def _temp_plots(self, x_star, x_i):

        fig, axs = plt.subplots(2)

        # Plot distributions of x_star and x_i to vlaidate selection
        # Plot 1
        plotting.gp_plot(axs[0], self.model, self.T_data, color="tab:blue")
        axs[0].set(title="Problem Model")

        plotting.gp_plot(axs[1], self.cost_model, self.T_cost, color="tab:blue")
        axs[1].set(title="Cost Model")

        plotting.plot_vline_axes(axs, x_star, color="red")
        plotting.plot_vline_axes(axs, x_i, color="green")

        plt.tight_layout()
        plt.show()
        input("Anything to continue")
