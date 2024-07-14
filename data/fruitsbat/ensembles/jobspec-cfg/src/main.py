from time import sleep
from log import Log
import schedule
import threading
import numio
from mpi4py import MPI
import daemon
from datetime import datetime
import os


def main():
    print(f"starting node {MPI.COMM_WORLD.Get_rank()} at {datetime.now()}")
    # start logging data about the current node
    log = Log()
    log.schedule(seconds_interval=int(os.environ["ENSEMBLES_LOG_EVERY_X_SECONDS"]))

    # check if this is the main job or a background daemon
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("found main node")
        # this is the main job
        numio_run = threading.Thread(target=numio.start)
        numio_run.start()
        while numio_run.is_alive():
            schedule.run_pending()
            sleep(1)
        print("main node finished, sending done signals...")
        numio.send_done_signals()
    else:
        print(f"found background node with rank: {MPI.COMM_WORLD.Get_rank()}")
        background_noise = threading.Thread(target=daemon.start)
        background_noise.start()
        while background_noise.is_alive():
            schedule.run_pending()
            sleep(1)

    # write out collected data
    log.write_histogram()
    log.write_json()
    print(f"shutting down node {MPI.COMM_WORLD.Get_rank()} at {datetime.now()}")


if __name__ == "__main__":
    main()
