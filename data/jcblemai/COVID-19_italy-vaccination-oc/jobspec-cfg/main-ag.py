import numpy as np
from ItalySetup import ItalySetupProvinces
from covidOCP import COVIDAgeStructuredOCP as COVIDAgeStructuredOCP
from covidOCP import COVIDParametersOCP
import pickle
import matplotlib.pyplot as plt
import click
import sys, os
from scenarios_utils import pick_scenario, build_scenario

nx, nc = 9, 3
states_names = ['S', 'E', 'P', 'I', 'A', 'Q', 'H', 'R', 'V']
when = 'future'
n_int_steps = 50
ocp = None


@click.command()
@click.option("-s", "--scenario_id", "scn_ids", default=1, help="Index of scenario to run")
@click.option("-n", "--nnodes", "nnodes", default=10, envvar="OCP_NNODES", help="Spatial model size to run")
@click.option("-t", "--ndays", "ndays", default=30, envvar="OCP_NDAYS", help="Number of days to run")
@click.option("--use_matlab", "use_matlab", envvar="OCP_MATLAB", type=bool, default=False, show_default=True,
              help="whether to use matlab for the current run")
@click.option("-a", "--objective", "objective", type=str, default='infection', show_default=True,
              help="Whether to use agestructured OCP")
@click.option("-f", "--file_prefix", "file_prefix", envvar="OCP_PREFIX", type=str, default='test',
              show_default=True, help="file prefix to add to identify the current set of runs.")
@click.option("-d", "--output_directory", "outdir", envvar="OCP_OUTDIR", type=str, default='model_output_AG/',
              show_default=True, help="Where to write runs")
@click.option("-o", "--optimize", "optimize", type=bool, default=False, show_default=True, help="Whether to optimize")
def cli(scn_ids, nnodes, ndays, use_matlab, objective, file_prefix, outdir, optimize):
    if not isinstance(scn_ids, list):
        scn_ids = [int(scn_ids)]
    return scn_ids, nnodes, ndays, use_matlab, objective, file_prefix, outdir, optimize

if __name__ == '__main__':
    # standalone_mode: so click doesn't exit, see
    # https://stackoverflow.com/questions/60319832/how-to-continue-execution-of-python-script-after-evaluating-a-click-cli-function
    scn_ids, nnodes, ndays, use_matlab, objective, file_prefix, outdir, optimize = cli(standalone_mode=False)
    os.makedirs(outdir, exist_ok=True)

    # All arrays here are (nnodes, ndays, (nx))
    setup = ItalySetupProvinces(nnodes, ndays, when)
    M = setup.nnodes
    N = setup.ndays - 1

    assert(objective == 'infection' or objective == 'death')

    if use_matlab:
        p = COVIDParametersOCP.OCParameters(setup=setup, M=M, when=when)
        if True:
            with open(f'italy-data/parameters_{nnodes}_{when}.pkl', 'wb') as out:
                pickle.dump(p, out, pickle.HIGHEST_PROTOCOL)
    else:
        with open(f'italy-data/parameters_{nnodes}_{when}.pkl', 'rb') as inp:
            p = pickle.load(inp)

    file_prefix = file_prefix

    for scn_id in scn_ids:
        scenario = pick_scenario(setup, scn_id)
        prefix = file_prefix + '-' + scenario['name']

        print(f"""Running scenario {scn_id}: {scenario['name']}, building setup with
        ndays: {ndays}
        nnodes: {nnodes}
        use_matlab: {use_matlab}
        when?  {when}
        rk_steps: {n_int_steps}
        obj: {objective}
        ---> Saving results to prefix: {prefix}""")

        p.apply_epicourse(setup, scenario['beta_mult'])

        control_initial = np.zeros((M, N, nc))

        results, state_initial, yell_death, yell_infection, mob = COVIDAgeStructuredOCP.integrate(N,
                                                                            setup=setup,
                                                                            parameters=p,
                                                                            controls=control_initial,
                                                                            save_to=f'{outdir}{prefix}-int-{nnodes}_{ndays}-nc',
                                                                            method='rk4',
                                                                            n_rk4_steps=n_int_steps)

        maxvaccrate_regional, delivery_national, stockpile_national_constraint, control_initial_all = build_scenario(setup, scenario, strategy=yell.sum(axis=1))
        control_initial = np.zeros((M, N, nc))*10

        if objective == 'infection':
            ag_id_to_vacc = 1
            cat_to_vacc = 'M'
        elif objective == 'death':
            ag_id_to_vacc = 2
            cat_to_vacc = 'O'

        unvac_nd = np.copy(setup.pop_node_ag[:,ag_id_to_vacc])* .7
        nv = results[results['cat'] == cat_to_vacc]
        incid = nv[nv['comp'].isin([f'yell_{objective}'])].groupby('place').sum()
        incid.sort_values('value', ascending=False, inplace=True)
        stockpile = 0
        for k in range(N):
            stockpile += delivery_national[k]
            for nodename in incid.sort_values('value', ascending=False).index:
                nd = setup.ind2name.index(nodename)
                to_allocate = maxvaccrate_regional[nd, k]
                to_allocate = min(to_allocate, maxvaccrate_regional[nd, k], unvac_nd[nd], stockpile)
                control_initial[nd, k, ag_id_to_vacc] = to_allocate
                stockpile -= to_allocate
                unvac_nd[nd] -= to_allocate

        results, state_initial, yell_death, yell_infection, mob = COVIDAgeStructuredOCP.integrate(N,
                                                                            setup=setup,
                                                                            parameters=p,
                                                                            controls=control_initial,
                                                                            save_to=f'{outdir}{prefix}-int-{nnodes}_{ndays}',
                                                                            n_rk4_steps=n_int_steps)

        ocp = COVIDAgeStructuredOCP.COVIDVaccinationOCP(N=N, n_int_steps=n_int_steps,
                                                        setup=setup, parameters=p,
                                                        objective=objective)

        ocp.update(parameters=p,
                   stockpile_national_constraint=stockpile_national_constraint,
                   maxvaccrate_regional=maxvaccrate_regional,
                   states_initial=state_initial,
                   control_initial=control_initial,
                   mob_initial=mob,
                   scenario_name=f'{outdir}{prefix}-{objective}-opt-{nnodes}_{ndays}')

        ocp.solveOCP()
