from themeda.transforms import ChipletBlock
from themeda.dataloaders import get_chiplets_list
from themeda.apps import get_dates, Interval
from pathlib import Path
import typer
from rich.progress import track

import torch.nn.functional as F

import torch

def get_mean_and_std(block,chiplets):
    value_sum, value_squared, total_timesteps = 0, 0, 0

    # for chiplet in chiplets[:100]:
    for chiplet in track(chiplets, "Chiplets:"):
        position = block.get_position(chiplet)
        
        values = block(chiplet)
        # breakpoint()

        value_sum += values.mean(dim=[1,2]).sum()
        value_squared += (values**2).mean(dim=[1,2]).sum()

        total_timesteps += values.shape[0]


    total_mean = value_sum/total_timesteps
    total_variance = (value_squared / total_timesteps) - (total_mean ** 2)
    total_std = torch.sqrt(total_variance)

    return total_mean, total_std


def main(
    chiplet_dir: Path, 
    max_chiplets:int=0,
    start:str=typer.Option("1988-01-01", help="The start date."),
    end:str=typer.Option("2018-01-01", help="The end date."),
    interval:Interval=typer.Option(Interval.YEARLY.value, help="The time interval to use."),
    max_years:int=0,
):
    chiplets = get_chiplets_list(chiplet_dir, max_chiplets)
    dates = get_dates(start=start, end=end, interval=interval)        

    block = ChipletBlock(base_dir=chiplet_dir, dates=dates, max_years=max_years, pad=False)

    persistence_total = 0.0
    pixel_count = 0

    # total_mean = 0.0
    # value_sum = 0.0
    # value_squared = 0.0
    # total_timesteps = 0


    mean, std = get_mean_and_std(block,chiplets)

    # with open("tmax-persistence.csv", "w") as f:
    #     print("subset", "id", "x", "y", "persistence_mean", sep=",", file=f)
    #     # for chiplet in chiplets[:100]:
    #     for chiplet in track(chiplets, "Chiplets:"):
    #         position = block.get_position(chiplet)
    #         # breakpoint()
    #         # breakpoint()
    #         t = block(chiplet)

    #         # After you've found the mean and standard deviation
    #         # We can normalise the value
    #         t = (t - mean)/std   

    #         # value_sum += values.mean(dim=[1,2]).sum()
    #         # value_squared += (values**2).mean(dim=[1,2]).sum()
    #         # total_timesteps += values.shape[0]


    #         # t = block(chiplet)
 
            
    #         time_t_plus_1 = t[1:]
    #         time_t = t[:-1]
    #         persistent = F.smooth_l1_loss(time_t, time_t_plus_1)

    #         persistence_mean = persistent.float().mean().item()
    #         persistence_total += persistence_mean
    #         print(chiplet.subset, chiplet.id, position[0], position[1], persistence_mean, sep=",", file=f, flush=True)

    # # chiplet_count = len(chiplets)
    # chiplet_count = 100

    # total_mean = value_sum/total_timesteps
    # total_variance = (value_squared / total_timesteps) - (total_mean ** 2)
    # total_std = torch.sqrt(total_variance)
    print("mean:", mean)
    print("std:", std)
    # print("persistence:", persistence_total/len(chiplets))

if __name__ == "__main__":
    typer.run(main)
