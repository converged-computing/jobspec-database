import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pygame as pg
import numpy as np
from Agent import Agent
from Detour_Agent import Detour_Agent
from Grid import Grid
import random
import argparse

GREY = (128, 128, 128)
WHITE = (255, 255, 255)

GREEN = (10, 196, 0)
RED =  (255, 0, 0)
ORANGE = (252, 127, 3)
BLUE = (30,144,255)

TEMP_AGENT_COLOUR = (242, 0, 255)

def isfinished(agents):
    finished = []
    agents_new = []

    for agent in agents:
        if agent.move():
            agents_new.append(agent)
        else:
            finished.append(agent)
    
    random.shuffle(agents_new)
    return finished, agents_new

def env_loop(grid: Grid, agents, p_dropoff = 0, vis=False, t=0, t_max=20000, round_density=2.3, alpha=0, speed=100, detours=False, test=False) -> None:
    '''runs simulation'''

    if vis:
        pg.init() # initialises imported pygame modules
        screen = pg.display.set_mode(grid.WINDOW_SIZE, pg.RESIZABLE)
        pg.display.set_caption("simple traffic simulation") # set window title
        clock = pg.time.Clock() # Used to manage how fast the screen updates

        # updates agents on screen
        move_event = pg.USEREVENT
        pg.time.set_timer(move_event, speed)
        max_pheromone = 0
        max_delay = 0
        orang_thresh = 9999
        red_thresh = 9999

        # -------- Main Game Loop ----------- #
        pause = False
        while t != t_max:
            pg.display.set_caption(f'[max_p = {round(max_pheromone, 2)}] [max delay = {max_delay}] [density = {round_density}] [alpha = {alpha}] [spread_decay = {p_dropoff}]')
            # HANDLE EVENTS
            for event in pg.event.get():
                # pause
                if event.type == pg.KEYDOWN and event.key == pg.K_p:
                    pause = not pause
                
                elif event.type == move_event and not pause:
                    
                    update_ph_list = []
                    _, agents = isfinished(agents=agents) # removes ones that have reached their destination 
                    # # calculate pheromone increase
                    for agent in agents:
                        update_ph_list.extend(agent.spread_pheromone())
                    # apply pheromone changes
                    for agent, update_val in update_ph_list:
                        agent.pheromone += update_val
                    # # add more agents
                    if not test: agents.extend(grid.generate_agents(round_density=round_density, p_dropoff=p_dropoff, alpha=alpha, detours=detours))
                    t += 1
                elif event.type == pg.MOUSEBUTTONDOWN:
                    pos = pg.mouse.get_pos()
                    column = pos[0] // (grid.CELL_SIZE + grid.MARGIN)
                    row = pos[1] // (grid.CELL_SIZE + grid.MARGIN)
                    print("Click ", pos, "Grid coordinates: ", row, column)
                elif event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                    pg.quit()
                elif event.type == pg.QUIT:  # If user clicked close
                    pg.quit()

            screen.fill(GREY)
            
            # draw grid
            for row in range(grid.CELLS_IN_WIDTH):
                for col in range(grid.CELLS_IN_HEIGHT):

                    colour = GREY
                    if grid.grid[row][col] in {"n","s","e", "w", "ne", "se", "nw", "sw"}: colour = WHITE # junction

                    # draw cells
                    pg.draw.rect(screen,
                                 colour,
                                 [(grid.MARGIN + grid.CELL_SIZE) * col + grid.MARGIN, # top y coord 
                                  (grid.MARGIN + grid.CELL_SIZE) * row + grid.MARGIN, # top x left
                                  grid.CELL_SIZE,   # width of rect
                                  grid.CELL_SIZE])  # height of rect
            # draw agents
            for agent in agents:
                if agent.delay > max_delay:
                    max_delay = agent.delay
                if max_pheromone < agent.pheromone:
                    max_pheromone = agent.pheromone
                    orang_thresh = 0.333 * max_pheromone
                    red_thresh = 0.666 * max_pheromone

                row, col = agent.grid_coord
                srow, scol = agent.src
                drow, dcol = agent.dst
                colour = GREEN
                if red_thresh > agent.pheromone >= orang_thresh:
                    colour = ORANGE
                elif agent.pheromone >= red_thresh:
                    colour = RED
                if type(agent.ID) == str:
                    colour=TEMP_AGENT_COLOUR

                if type(agent.ID) == int:
                    pg.draw.rect(screen, # entrance
                                BLUE, 
                                [(grid.MARGIN + grid.CELL_SIZE) * scol + grid.MARGIN, # top y coord 
                                (grid.MARGIN + grid.CELL_SIZE) * srow + grid.MARGIN, # top x left
                                grid.CELL_SIZE,   # width of rect
                                grid.CELL_SIZE])  # height of rect
                    
                    pg.draw.rect(screen, # exit
                                RED, 
                                [(grid.MARGIN + grid.CELL_SIZE) * dcol + grid.MARGIN, # top y coord 
                                (grid.MARGIN + grid.CELL_SIZE) * drow + grid.MARGIN, # top x left
                                grid.CELL_SIZE,   # width of rect
                                grid.CELL_SIZE])  # height of rect

                pg.draw.rect(screen, # agent
                            colour, 
                            [(grid.MARGIN + grid.CELL_SIZE) * col + grid.MARGIN, # top y coord 
                             (grid.MARGIN + grid.CELL_SIZE) * row + grid.MARGIN, # top x left
                             grid.CELL_SIZE,   # width of rect
                             grid.CELL_SIZE])  # height of rect
            
            clock.tick(120) # fps
            pg.display.flip() # draws new frame
        pg.quit()

    else:
        while t != t_max:
            update_ph_list = []
            finished, agents = isfinished(agents=agents)
            num_of_finished = len(finished)

            # calculate pheromone increase
            for agent in agents:
                update_ph_list.extend(agent.spread_pheromone())
            # apply pheromone changes
            for agent, update_val in update_ph_list:
                agent.pheromone += update_val
            # add more agents
            agents.extend(grid.generate_agents(round_density=round_density, p_dropoff=p_dropoff, alpha=alpha, detours=detours)) # fixme need to account for new weights on pheromone and distances
            t += 1
            
            if finished:
                min_delay = min(agent.delay for agent in finished)
                max_delay = max(agent.delay for agent in finished)
                mean_delay = np.mean([agent.delay for agent in finished])
                print(min_delay, max_delay, mean_delay, num_of_finished)
            else:
                print(0, 0, 0, num_of_finished)


parser = argparse.ArgumentParser()
parser.add_argument("-vis", action="store_true")
parser.add_argument("-detours", action="store_true")
parser.add_argument("-test", action="store_true")
parser.add_argument("-density", default=2.3, type=float)
parser.add_argument("-alpha", default=0, type=int)
parser.add_argument("-t_max", default=20000, type=int)
parser.add_argument("-roads", default=5, type=int)
parser.add_argument("-speed", default=100, type=int)
parser.add_argument("-p_dropoff", default=1, type=float)
args = parser.parse_args()


if not args.vis:
    grid = Grid(num_roads_on_axis = args.roads)

    agent = grid.generate_agents(round_density=args.density, 
                                 alpha=args.alpha, 
                                 p_dropoff=args.p_dropoff,
                                 detours=args.detours)

    env_loop(grid=grid, 
             round_density=args.density, 
             alpha=args.alpha, 
             t_max=args.t_max, 
             agents=agent, 
             vis=False, 
             p_dropoff=args.p_dropoff,
             detours=args.detours) 
else:
    if args.test:
        args.roads = 3
        args.test = True

    grid = Grid(num_roads_on_axis = args.roads)
    agent = grid.generate_agents(round_density=args.density, 
                                 alpha=args.alpha, 
                                 p_dropoff=args.p_dropoff,
                                 detours=args.detours,
                                 test=args.test)
    
    # if args.test:
    #     agent.append(Detour_Agent(ID="tracker", src=(32, 16, "n"), grid=grid, move_buffer=["e"]))
    #     agent.append(Detour_Agent(ID="tracker", src=(32, 33, "n"), grid=grid, move_buffer=["e"], pheromone=0))

    env_loop(grid=grid, 
             round_density=args.density, 
             alpha=args.alpha, 
             t_max=args.t_max, 
             agents=agent, 
             vis=True, 
             p_dropoff=args.p_dropoff,
             speed=args.speed,
             detours=args.detours,
             test=args.test)

# python main.py -vis -density 3 -alpha 0 -spread_decay 0.00