problems = [
    ("Forrester", {}),
    ("Levy", {}),
    ("Hartmann6", {})
]

methods = {
    #"EI": "EI",
    #"UCB": "UCB",
    #"EICostRatio": "EICostRatio",
    #"UCBCostRatio": "UCBCostRatio"
    #"HardLocalPenalisationBatchBOCost": "HardLocalPenalisationBatchBOCost",
    "LocalPenalisationBatchBOCost": "LocalPenalisationBatchBOCost"
}

bo_names = {
    #"EI": "AsyncBO",
    #"UCB": "AsyncBO",
    #"EICostRatio": "AsyncCostAcqBO",
    #"UCBCostRatio": "AsyncCostAcqBO",
    "HardLocalPenalisationBatchBOCost": "AsyncSKBO",
    "LocalPenalisationBatchBOCost": "AsyncSKBO"
}

killing_names = [
    "NoKilling",
    "DeterministicKilling",
    "ProbabilisticKilling"
]

acq_params = {
    "n_opt_samples": 1 * 1000,
    "n_opt_bfgs": 10,
    "acq_name": "EI",
}

killing_params = {
    "NoKilling": {
        "n_opt_samples": 1 * 1000,
        "n_opt_bfgs": 10,
    },
    "DeterministicKilling": {
        "delta": 1,
        "acq_name": "",
        "acq_params": acq_params,
        "n_opt_samples": 1 * 1000,
        "n_opt_bfgs": 10,
    },
    "ProbabilisticKilling": {
        "alpha": 0.8,
        "epsilon": 1e-8,
        "n_opt_samples": 1 * 1000,
        "n_opt_bfgs": 10,
    }
}

time_functions = [
    ("corrtime", "job-dependant"),
    ("negcorrtime", "job-dependant"),
    ("consttime", "job-dependant"),
]

max_steps = 100
num_runs = 21
workers = [2, 4, 8]



import numpy as np
from aegis.util import generate_save_filename
from aegis.gen_training_data import generate_training_data_LHS
from aegis.optim import perform_optimisation

for problem_name, problem_params in problems:

    # Initial point generation
    generate_training_data_LHS(problem_name, n_exp_start=1, n_exp_end=num_runs)

    for time_name, interface in time_functions:

        for n_workers in workers:

            for acq_name in methods:

                bo_name = bo_names[acq_name]

                for kill_name in killing_names:

                    kill_params = killing_params[kill_name]

                    if "acq_name" in kill_params:
                        kill_params["acq_name"] = acq_name

                    for run in range(num_runs):

                        print(generate_save_filename(
                            time_name,
                            problem_name,
                            max_steps,
                            n_workers,
                            acq_name,
                            run+1,
                            bo_name,
                            kill_name,
                            problem_params,
                            acq_params,
                            None,
                        ))

                        perform_optimisation(
                            problem_name=problem_name,
                            problem_params=problem_params,
                            run_no=run+1,
                            budget=max_steps,
                            n_workers=n_workers,
                            acq_name=acq_name,
                            acq_params=acq_params,
                            time_name=time_name,
                            save_every=10,
                            repeat_no=None,
                            bo_name=bo_name,
                            kill_name=kill_name,
                            killing_params=kill_params,
                        )