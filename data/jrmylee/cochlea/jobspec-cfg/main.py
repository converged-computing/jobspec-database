from cochlea import generate_spectrograms_from_ds
from transformations.room import get_random_room, get_room_impulse, convolve_with_room
import json
from types import SimpleNamespace
import random

def get_room_irs(num_rooms=50):
    print("getting room irs")
    rooms = []
    for i in range(num_rooms):
        room = get_random_room()
        ir = get_room_impulse(room)
        rooms.append(ir)
    return rooms

irs = get_room_irs()

def room_aug(x):
    ir = random.choice(irs)
    return convolve_with_room(x, ir)

print("beginning preprocess")
with open("params.json") as file:
    hparams = json.load(file, object_hook=lambda d: SimpleNamespace(**d))
    generate_spectrograms_from_ds(
    ds_path=hparams.ds_path, 
    mapping_filename="maestro-v3.0.0.csv", 
    save_path=hparams.save_path, 
    augment_fn=room_aug,
    hparams=hparams)
