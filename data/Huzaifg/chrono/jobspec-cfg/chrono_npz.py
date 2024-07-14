import numpy as np
import sys
from pathlib import Path
import pandas as pd

train_output = {}
test_output = {}
vel_mean = np.zeros((3), dtype=np.float32)
vel_std = np.zeros((3), dtype=np.float32)
acc_mean = np.zeros((3), dtype=np.float32)
acc_std = np.zeros((3), dtype=np.float32)
cumulative_sum_vel = np.zeros((3), dtype=np.float32)
cumulative_sum_acc = np.zeros((3), dtype=np.float32)
cumulative_sumsq_vel = np.zeros((3), dtype=np.float32)
cumulative_sumsq_acc = np.zeros((3), dtype=np.float32)
cumulative_count = 0
sims = 0
nmax = 350
dt = 1

def convert(group, file_num, folder_input, output, train) -> None:
    global cumulative_count, sims, vel_mean, vel_std, acc_mean, acc_std, cumulative_sum_vel, cumulative_sum_acc, cumulative_sumsq_acc, cumulative_sumsq_vel

    try:
        if (group < 5):
            boundary = pd.read_csv(f"{folder_input}BCE_Rigid0.csv", header="infer", delimiter=",")
            boundary = boundary.to_numpy(dtype=np.float32)
            boundary = boundary[:, :3]
            boundary[:, [2, 1]] = boundary[:, [1, 2]]
            bce_lines = boundary.shape[0]
        else:
            bce_lines = 0

        # Only to create positions array
        sph_f = pd.read_csv(f"{folder_input}fluid0.csv", header="infer", delimiter=",")
        sph = sph_f.to_numpy(dtype=np.float32)
        sph_lines = sph.shape[0]
        positions = np.empty((nmax, bce_lines + sph_lines, 3), dtype=np.float32)

        for i in range(nmax):
            sph_f = pd.read_csv(f"{folder_input}fluid{i}.csv", header="infer", delimiter=",")
            sph = sph_f.to_numpy(dtype=np.float32)
            # Only need ground truth position
            sph = sph[:, :3]
            # Swap y, z is intentional 
            sph[:, [2, 1]] = sph[:, [1, 2]]
            
            # Get all positions first
            if (group < 5):
                positions[i, :, :] = np.concatenate((boundary, sph))
            else:
                positions[i, :, :] = sph
                
        bce_particle_num = np.full((bce_lines), 3, dtype=int)
        sph_particle_num = np.full((sph_lines), 6, dtype=int)
        particle_num = np.concatenate((bce_particle_num, sph_particle_num))

        output[f"{group}_simulation_trajectory_{file_num}"] = (positions, particle_num)
        
        if (train):
            velocity = positions[:, bce_lines:, :].copy()
            velocity[1:] = (positions[1:, bce_lines:, :] - positions[:-1, bce_lines:, :]) / dt
            velocity[0] = 0
            flat_velocity = np.reshape(velocity, (-1, 3))

            acceleration = (velocity[1:] - velocity[:-1]) / dt
            acceleration[0] = 0
            flat_acceleration = np.reshape(acceleration, (-1, 3))

            # Yongjin
            cumulative_count += len(flat_velocity)

            cumulative_sum_vel += np.sum(flat_velocity, axis=0)
            cumulative_sum_acc += np.sum(flat_acceleration, axis=0)

            cumulative_sumsq_vel += np.sum(np.square(flat_velocity), axis=0)
            cumulative_sumsq_acc += np.sum(np.square(flat_acceleration), axis=0)

            vel_mean = cumulative_sum_vel / cumulative_count
            acc_mean = cumulative_sum_acc / cumulative_count
            vel_std = np.sqrt((cumulative_sumsq_vel / cumulative_count - np.square(cumulative_sum_vel / cumulative_count)))
            acc_std = np.sqrt((cumulative_sumsq_acc / cumulative_count - np.square(cumulative_sum_acc / cumulative_count)))

        sims += 1
        print(f"Finished {folder_input}")
    except:
        print(f"Error at {folder_input}. Ignoring this simulation trajectory and moving on.")

train_split = int(sys.argv[1])
folder = sys.argv[2]
DEMO_PARENT = "/work/09874/tliangwi/ls6/DEMO_OUTPUT/"

save_dir = f"/work/09874/tliangwi/ls6/{folder}/"
dataset_dir = f"{save_dir}datasets/"
Path(dataset_dir).mkdir(exist_ok=True, parents=True)
Path(f"{save_dir}models").mkdir(exist_ok=True)
Path(f"{save_dir}output").mkdir(exist_ok=True)

for bs in range(1, train_split + 1):
    for group in range(1, 5 + 1):
        convert(group, bs, f"{DEMO_PARENT}{group}_BAFFLE_FLOW_TRAIN_{bs}/particles/", train_output, True)

for bs in range(train_split + 1, 200 + 1):
    for group in range(1, 5 + 1):
        convert(group, bs, f"{DEMO_PARENT}{group}_BAFFLE_FLOW_TRAIN_{bs}/particles/", test_output, False)

print(f"VELOCITY MEAN: {vel_mean}")
print(f"ACCELERATION MEAN: {acc_mean}")
print(f"VELOCITY VARIANCE (Use sqrt): {vel_std}")
print(f"ACCELERATION VARIANCE (Use sqrt): {acc_std}")
print(f"TRAINING SIMULATIONS COMPLETED: {sims}")
print(f"TOTAL LINES: {cumulative_count}")

train_npz_output = f"{dataset_dir}train.npz"
np.savez_compressed(train_npz_output, **train_output)
test_npz_output = f"{dataset_dir}test.npz"
np.savez_compressed(test_npz_output, **test_output)
