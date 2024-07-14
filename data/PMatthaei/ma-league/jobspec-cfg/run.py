#!/usr/bin/env python3
import re
import subprocess
from typing import Dict


args = []

#
# !! WARN !! No imports from src/** allowed !!
# Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library.
# Try to import numpy first or set the threading layer accordingly. Set MKL_SERVICE_FORCE_INTEL to force it.
#

def options_string(options: Dict):
    return ", ".join([f"{num}) {option}" for num, option in options.items()])


def select(title, options, arg, only_arg=False, is_enum=False):
    response = None
    options.append("Quit")
    num_options = {str(num + 1): option for num, option, in enumerate(options)}
    while response not in options:
        response = input(f"----- Choose {title}:\n {options_string(num_options)} \n Enter: ")
        response = num_options[response]
        if response == "Quit":
            print("Quitting Assistent")
            exit(1)
        else:
            yes = True if response == "True" else False
            print(f"\n Choice: {response} \n")

            if yes and only_arg:
                args.append(f"{arg}")
                return yes
            elif not only_arg:
                response = response if yes or is_enum else response.lower()
                args.append(f"{arg}={response}")
                return yes


def enter(title, arg):
    response = input(f"----- Enter desired {title}: \n ")
    print(f"\n Choice: {response} \n")
    args.append(f"{arg}={response}")


def choice(title, arg, only_arg=False) -> bool:
    yes = select(title=title, options=["True", "False"], arg=arg, only_arg=only_arg)
    return yes

def confirm():
    _ = input("Confirm the command via any key.")

if __name__ == '__main__':
    print("Starting Experiment Assistant...")
    python_cmd_base = "python src/central_worker_main.py"

    select(title="MARL algorithm Config", options=["QMIX", "VDN", "IQL"], arg="--config")
    select(title="Environment Config", options=["ma"], arg="--env-config")
    select(title="League Config", options=["Ensemble", "Matchmaking", "Rolebased"], arg="--league-config")
    select(title="Experiment Config", options=["Ensemble", "Matchmaking", "Rolebased"], arg="--experiment")
    select(title="Matchmaking Config", options=["Adversaries", "FSP", "PFSP"], arg="--matchmaking")

    enter("League size", arg="--league_size")
    enter("Team size", arg="--team_size")

    use_cuda = choice("CUDA", arg="--use_cuda")
    choice("CUDA work balance", arg="--balance-cuda-workload", only_arg=True) if use_cuda else None
    choice("Tensorboard", arg="--use_tensorboard")

    save_model = choice("model saving", arg="--save_model")
    enter("model saving interval", arg="--save_model_interval") if save_model else None

    force_unit = choice("unit enforcing", arg="force-unit", only_arg=True)
    if force_unit:
        choice("unique forced unit in team", arg="--unique", only_arg=True)
        select(title="role of enforced unit", options=["TANK", "HEALER", "ADC"], arg="--role", is_enum=True)
        select(title="attack type of enforced unit", options=["RANGED", "MELEE"], arg="--attack", is_enum=True)
    else:
        print("Currently no support for non-unit enforcing!")
        exit(0)
    python_cmd = f"{python_cmd_base} {' '.join(args)}"

    python_cmd = re.sub(' +', ' ', python_cmd)  # Clean multiple whitespaces
    print(f"\n\n  Command: {python_cmd} \n\n")

    confirm()

    subprocess.check_call(python_cmd.split(" "))

    exit(1)
