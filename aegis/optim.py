import os
import torch
import numpy as np
from . import gp, test_problems, transforms, util, executor, time_dists, batch, killing


def perform_optimisation(
    problem_name,
    problem_params,
    run_no,
    budget,
    n_workers,
    acq_name,
    acq_params,
    time_name,
    save_every=10,
    repeat_no=None,
    bo_name='AsyncBO',
    kill_name=None,
    killing_params=None,
):

    if kill_name is None:
        kill_name = "NoKilling"

    # set up the saving paths
    save_path = util.generate_save_filename(
        time_name,
        problem_name,
        budget,
        n_workers,
        acq_name,
        run_no,
        bo_name,
        kill_name,
        problem_params,
        acq_params,
        killing_params,
        repeat_no=repeat_no,
    )

    if os.path.exists(save_path):
        load_path = save_path
        print("Loading saved run")
    else:
        load_path = util.generate_data_filename(
            problem_name, run_no, problem_params, repeat_no=repeat_no
        )

    # load the training data
    data = torch.load(load_path)
    Xtr = data["Xtr"]
    Ytr = data["Ytr"]

    # if it has additional arguments add them to the dictionary passed to f
    if "problem_params" in data:
        problem_params.update(data["problem_params"])

    print(f"Training data shape: {Xtr.shape}")

    # load the problem instance
    f = getattr(test_problems, problem_name)(**problem_params)

    # wrap the problem for torch and so that it resides in [0, 1]^d
    f = util.TorchProblem(util.UniformProblem(f))

    # instantiate the time function
    time_class = getattr(time_dists, time_name)

    # get the acquisition function class
    acq_class = getattr(batch, acq_name)

    # get the BO class
    asbo = eval(bo_name)

    # run the BO
    if kill_name is not None:
        asbo = asbo(
            f,
            Xtr,
            Ytr,
            acq_class,
            acq_params,
            budget,
            n_workers,
            time_class,
            q=1,
            verbose=True,
            kill_name=kill_name,
            killing_params=killing_params
        )
    else:
        asbo = asbo(
            f,
            Xtr,
            Ytr,
            acq_class,
            acq_params,
            budget,
            n_workers,
            time_class,
            q=1,
            verbose=True,
        )

    # useful stuff to keep a record of
    save_dict = {
        "problem_name": problem_name,
        "problem_params": problem_params,
        "q": 1,
        "n_workers": n_workers,
        "acq_name": acq_name,
        "time_name": time_name,
        "budget": budget,
    }

    while not asbo.finished:
        asbo.step()

        # save every so often or when we've finished
        if asbo.Xtr.shape[0] % save_every == 0 or asbo.finished:
            # get the results so far
            res = asbo.get_results()
            save_dict.update(res)

            torch.save(obj=save_dict, f=save_path)
            print(f"Saving: {save_path:s}")

    print("Finished run")

class AsyncBO:
    def __init__(
        self,
        func,
        Xtr,
        Ytr,
        acq_class,
        acq_params,
        budget,
        n_workers,
        time_func,
        q=1,
        verbose=False,
    ):

        self.f = func
        self.Xtr = Xtr
        self.Ytr = Ytr
        self.acq_class = acq_class
        self.acq_params = acq_params
        self.budget = budget
        self.n_workers = n_workers
        self.q = q
        self.output_transform = getattr(transforms, "Transform_Standardize",)
        self.time_func = self._init_time_func(time_func)
        self.verbose = verbose

        self.interface = executor.SimExecutorJumpToCompletedJob(
            n_workers=n_workers, time_func=time_func, verbose=verbose
        )

        self.dtype = Ytr.dtype

        self.ue = util.UnderEval(self.n_workers, self.f.dim, self.dtype)

        # time storage
        self.time = torch.zeros_like(Ytr, dtype=self.dtype)
        self.time_taken = torch.zeros_like(Ytr, dtype=self.dtype)

        # counters
        self.n_submitted = Xtr.shape[0]
        self.n_completed = Xtr.shape[0]

        # current iteration number
        self.init_size = Xtr.shape[0]

        # bounds
        self.ls_bounds = [1e-4, np.sqrt(Xtr.shape[1])]
        self.out_bounds = [1e-4, 10]

        # acquisition function, initialised after we have built a model
        self._acq = None

    def _init_time_func(self, time_class):
        return time_class()

    @property
    def acq(self):
        if self._acq is None:
            self._acq = self.acq_class(
                model=self.model,
                lb=self.f.lb,
                ub=self.f.ub,
                under_evaluation=self.ue.get(),
                **self.acq_params,
            )

        return self._acq

    @property
    def finished(self):
        return self.n_completed == self.budget

    def _process_completed_jobs(self):
        # get the completed jobs
        completed_jobs = self.interface.get_completed_jobs()
        n = len(completed_jobs)

        # storage, so we only cat once
        _Xtr = torch.zeros((n, self.f.dim), dtype=self.dtype)
        _Ytr = torch.zeros(n, dtype=self.dtype)
        _time_taken = torch.zeros(n, dtype=self.dtype)
        _time = torch.zeros(n, dtype=self.dtype)

        # add the results to the training data
        for i, job in enumerate(completed_jobs):
            _Xtr[i] = job["x"]
            _Ytr[i] = job["y"]
            _time_taken[i] = torch.as_tensor(job["t"], dtype=self.dtype)
            _time[i] = self.interface.status["t"]

            # remove from under evaluation
            self.ue.remove(_Xtr[i])

            if self.verbose:
                print(
                    f"Completed -> f(x): {_Ytr[i]:0.3f}",
                    f"time taken: {_time_taken[i]:0.3f}",
                )

        # store the results
        self.Xtr = torch.cat((self.Xtr, _Xtr))
        self.Ytr = torch.cat((self.Ytr, _Ytr))
        self.time_taken = torch.cat((self.time_taken, _time_taken))
        self.time = torch.cat((self.time, _time))

        # add to the completed jobs counter
        self.n_completed += len(completed_jobs)

    def _update_acq(self):
        self.acq.update(self.model, self.ue.get())

    def _create_job(self, x):
        return {"x": x, "f": self.f}

    def _adopt_x(self, x):

        # create the job and submit it
        job = self._create_job(x)
        self.interface.add_job_to_queue(job)

        # add to under evaluation
        self.ue.add(x)

        # Train model gp
        self._setup_model_gp()

        # Train cost gp
        self._setup_cost_gp()

        if self.verbose:
            print("Submitted ->", x.numpy().ravel())

    def _create_and_submit_jobs(self):
        # only submit up to the budget
        n_to_submit = np.minimum(
            self.budget - self.n_submitted,  # total left to submit
            self.interface.status["n_free_workers"],  # free workers
        )
        if n_to_submit < 1:
            return

        # update the acquisition function ready for use.
        self._update_acq()

        # get locations to evaluate from the acquisition function, create a
        # job, and submit it
        for _ in range(n_to_submit):
            # get the next location to evaluate
            x = self.acq.get_next()

            # Adopt x
            self._adopt_x(x)

        self.n_submitted += n_to_submit

    def _setup_step(self):
        if self.verbose:
            print(
                f" Submitted: {self.n_submitted} / {self.budget}"
                f" Completed: {self.n_completed} / {self.budget}"
            )

        # if the optimisation is complete, do nothing.
        if self.finished:
            print("finished")
            return True

        # run the jobs until we there's one free
        if self.n_submitted < self.budget:
            self.interface.run_until_n_free(self.q)

        # if we've submitted our budget's worth of jobs, keep waiting
        # until one more worker each time is free. i.e. 1, 2, ..., n_workers
        else:
            self.interface.run_until_n_free(
                self.n_workers - (self.budget - self.n_completed) + 1
            )

        if self.verbose:
            print("Time:", self.interface.status["t"])

        # get the completed jobs
        self._process_completed_jobs()

        # if we've submitted our budget's worth of jobs,
        # do not submit any more
        if self.n_submitted == self.budget:
            return True

        # If we've got here we are setup and do
        # not need to end this step
        return False

    def _setup_gp(self, train_x, train_y):

        # Scale outputs
        T = self.output_transform(train_y)
        train_y = T.scale_mean(train_y)

        # Fit gp
        model, likelihood = gp.create_and_fit_GP(
            train_x=train_x,
            train_y=train_y,
            ls_bounds=self.ls_bounds,
            out_bounds=self.out_bounds,
            n_restarts=10,
            verbose=self.verbose
        )

        return model, likelihood

    def _setup_model_gp(self):
        self.model, self.likelihood = self._setup_gp(self.Xtr, self.Ytr)
        self.model.eval()

    def step(self):
        
        # Setup the step
        skip_step = self._setup_step()

        # If we need to end this step, do
        if skip_step:
            return

        # Train model gp
        self._setup_model_gp()

        # submit jobs
        self._create_and_submit_jobs()

        if self.verbose:
            print("------------------------------")
            print()

    def get_results(self):
        resd = {
            "Xtr": self.Xtr,
            "Ytr": self.Ytr,
            "time_taken": self.time_taken,
            "time": self.time,
        }

        return resd

    def get_models(self):
        resd = {"ProblemModel": self.model}
        return resd

class AsyncDependantCostBO(AsyncBO):
    def __init__(
        self,
        func,
        Xtr,
        Ytr,
        acq_class,
        acq_params,
        budget,
        n_workers,
        time_func,
        q=1,
        verbose=False,
    ):

        super().__init__(
            func,
            Xtr,
            Ytr,
            acq_class,
            acq_params,
            budget,
            n_workers,
            time_func,
            q=q,
            verbose=verbose,
        )

    def _create_job(self, x):
        return {"x": x, "f": self.f, "t": self.time_func(x)}

class AsyncProblemDependantCost(AsyncDependantCostBO):
    def __init__(
        self,
        func,
        Xtr,
        Ytr,
        acq_class,
        acq_params,
        budget,
        n_workers,
        time_func,
        q=1,
        verbose=False,
    ):

        super().__init__(
            func,
            Xtr,
            Ytr,
            acq_class,
            acq_params,
            budget,
            n_workers,
            time_func,
            q=q,
            verbose=verbose,
        )

    def _init_time_func(self, time_class):
        return time_class(self.f)

class AsyncCostAcqBO(AsyncProblemDependantCost):
    def __init__(
        self,
        func,
        Xtr,
        Ytr,
        acq_class,
        acq_params,
        budget,
        n_workers,
        time_func,
        q=1,
        verbose=False,
    ):

        super().__init__(
            func,
            Xtr,
            Ytr,
            acq_class,
            acq_params,
            budget,
            n_workers,
            time_func,
            q=q,
            verbose=verbose,
        )

        self.cost = torch.zeros_like(Ytr, dtype=self.dtype)

        # sample time function for time of initialised points Xtr
        for i in range(self.time_taken.shape[0]):
                self.cost[i] = torch.as_tensor(self.time_func(Xtr[i]))

    @property
    def acq(self):
        if self._acq is None:
            self._acq = self.acq_class(
                model=self.model,
                cost_model=self.cost_model,
                T_data=self.output_transform(self.Ytr),
                T_cost=self.output_transform(self.cost),
                lb=self.f.lb,
                ub=self.f.ub,
                under_evaluation=self.ue.get(),
                **self.acq_params,
            )

        return self._acq

    def _update_acq(self):
        self.acq.update(self.model, self.ue.get(), self.cost_model)

    def _finished_job_cost(self, job):
        return torch.as_tensor(job["t"], dtype=self.dtype)

    def _process_completed_jobs(self):
        # get the completed jobs
        completed_jobs = self.interface.get_completed_jobs()
        n = len(completed_jobs)

        # storage, so we only cat once
        _Xtr = torch.zeros((n, self.f.dim), dtype=self.dtype)
        _Ytr = torch.zeros(n, dtype=self.dtype)
        _time_taken = torch.zeros(n, dtype=self.dtype)
        _time = torch.zeros(n, dtype=self.dtype)
        _cost = torch.zeros(n, dtype=self.dtype)

        # add the results to the training data
        for i, job in enumerate(completed_jobs):
            _Xtr[i] = job["x"]
            _Ytr[i] = job["y"]
            _time_taken[i] = torch.as_tensor(job["t"], dtype=self.dtype)
            _time[i] = self.interface.status["t"]
            _cost[i] = self._finished_job_cost(job)

            # remove from under evaluation
            self.ue.remove(_Xtr[i])

            if self.verbose:
                print(
                    f"Completed -> f(x): {_Ytr[i]:0.3f}",
                    f"time taken: {_time_taken[i]:0.3f}",
                )

        # store the results
        self.Xtr = torch.cat((self.Xtr, _Xtr))
        self.Ytr = torch.cat((self.Ytr, _Ytr))
        self.time_taken = torch.cat((self.time_taken, _time_taken))
        self.time = torch.cat((self.time, _time))
        self.cost = torch.cat((self.cost, _cost))

        # add to the completed jobs counter
        self.n_completed += len(completed_jobs)

    def _setup_cost_gp(self):
        self.cost_model, self.cost_likelihood = self._setup_gp(self.Xtr, self.cost)
        self.cost_model.eval()

    def step(self):

        # Setup the step
        skip_step = self._setup_step()

        # If we need to end this step, do
        if skip_step:
            return

        # Train model gp
        self._setup_model_gp()

        # Train cost gp
        self._setup_cost_gp()

        # submit jobs
        self._create_and_submit_jobs()

        if self.verbose:
            print("------------------------------")
            print()

    def get_results(self):
        resd = {
            "Xtr": self.Xtr,
            "Ytr": self.Ytr,
            "time_taken": self.time_taken,
            "time": self.time,
            "cost": self.cost,
        }

        return resd

    def get_models(self):
        resd = {
            "ProblemModel": self.model,
            "CostModel": self.cost_model,
            }
        return resd

class AsyncSKBO(AsyncCostAcqBO):
    def __init__(
        self,
        func,
        Xtr,
        Ytr,
        acq_class,
        acq_params,
        budget,
        n_workers,
        time_func,
        q=1,
        verbose=False,
        kill_name=None,
        killing_params=None,
    ):

        super().__init__(
            func,
            Xtr,
            Ytr,
            acq_class,
            acq_params,
            budget,
            n_workers,
            time_func,
            q=q,
            verbose=verbose,
        )

        # Get class handle for killing method
        self._killing_method_class = getattr(killing, kill_name)
        self.killing_params = killing_params

        # Initialise after we have built a model
        self._killing_method = None

        self._max_to_kill = self.n_workers

    def _get_ongoing_run_times(self):
        ongoing_tasks = self.interface._running_tasks
        res = [0] * len(ongoing_tasks)
        times = self.interface._ongoing_start_times
        for k in range(len(ongoing_tasks)):
            res[k] = self.interface._time_ticker - times[k]
        return res

    @property
    def killing_method(self):
        if self._killing_method is None:
            self._killing_method = self._killing_method_class(
                model=self.model,
                cost_model=self.cost_model,
                T_data=self.output_transform(self.Ytr),
                T_cost=self.output_transform(self.cost),
                lb=self.f.lb,
                ub=self.f.ub,
                under_evaluation=self.ue.get(),
                eval_times=self._get_ongoing_run_times(),
                **self.killing_params,
            )

        return self._killing_method

    def _update_killing_method(self):
        self.killing_method.update(self.model, self.ue.get(), self.cost_model, self._get_ongoing_run_times())

    def kill_x(self, x):

        # Increment killed counter
        self._killed += 1

        # Remove from UnderEval
        self.ue.remove(x)

        # Remove from interface
        self.interface.kill_evaluation(x)

        if self.verbose:
            print("Killing ->", x.numpy().ravel())

    def _kill_and_submit_jobs(self):
        # only submit up to the budget
        n_to_submit = np.minimum(
            self.budget - self.n_submitted,  # total left to submit
            self.interface.status["n_free_workers"],  # free workers
        )
        if n_to_submit < 1:
            return

        # Setup killed evaluations counter
        self._killed = 0

        # update the acquisition function ready for use
        self._update_acq()

        # update killing function ready for use
        self._update_killing_method()

        # get locations to evaluate from the acquisition function, create a
        # job, and submit it
        for _ in range(n_to_submit):

            # get the next location to evaluate
            x_star = self.acq._get_next()

            if _ == n_to_submit - 1:
                # While we have not finished killing
                killed_something = True
                while killed_something: 

                    # If we have killed our max per iteration
                    # break out of killing loop
                    if self._killed >= self._max_to_kill:
                        break

                    # Determine x to kill, if any
                    x_i = self.killing_method._get_next(x_star)

                    # If something to kill
                    if x_i is not None:
                        #kill x_i
                        self.kill_x(x_i)
                        #adopt x_star
                        self._adopt_x(x_star)
                        #generate new x_star
                        self._update_acq()
                        self._update_killing_method()
                        x_star = self.acq._get_next()
                    else:
                        # We haven't killed anything, so we're done for now
                        killed_something = False

            # We have checked all evaluations and do not want to kill them
            # so adopt x_star
            self._adopt_x(x_star)
            self._update_acq()
            self._update_killing_method()

        self.n_submitted += n_to_submit

    def step(self):

        # Setup the step
        end_step = self._setup_step()

        # If we need to end this step, do
        if end_step:
            return

        # Train model gp
        self._setup_model_gp()

        # Train cost gp
        self._setup_cost_gp()

        # submit jobs
        self._kill_and_submit_jobs()

        if self.verbose:
            print("------------------------------")
            print()
