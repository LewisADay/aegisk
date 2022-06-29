
def get_res_filename(
    time_func_name,
    problem_name,
    budget,
    n_workers,
    acq_name,
    run_no,
    problem_params,
    acq_params,
    repeat_no,
):
    _tmp = ""
    keys = list(problem_params.keys())
    keys = keys.sort()
    for key in keys:
        _tmp += f"_{key:s}{problem_params[key]:s}"

    
