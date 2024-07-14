'''
Driver file for our environment, run this file first!

'''
import sys
import os
import argparse
import time
from xml.parsers.expat import model
import numpy as np
import pybullet as p
import pybullet_data
from cbs import cbs
from PRIMAL.primal_testing import make_name, run_simulations, PRIMAL
import matplotlib.pyplot as plt

from utils import BLUE, GREEN, YELLOW, Point, change_color, create_box, set_point, wait_for_duration, RED, GREY, \
    get_point, set_euler, get_euler, joints_from_names, set_joint_positions, RGBA, set_preview, create_sphere

MAP_SIZE = 10
# BASE_JOINTS = ['x', 'y', 'theta']
BASE_JOINTS = ['x', 'y']


def creat_walls(size=MAP_SIZE, wall_width=0.2):
    '''
    Input: Size of the MAP and the width of the walls.
    Creates the walls around our environment. 
    returns: an array containing the walls indexes.
    '''
    # print(f"CREATING WALLS FOR MAP OF SIZE: {size}")

    wall1 = create_box(size + wall_width, wall_width, wall_width, color=GREY)
    set_point(wall1, Point(y=size / 2., z=wall_width / 2.))
    wall2 = create_box(size + wall_width, wall_width, wall_width, color=GREY)
    set_point(wall2, Point(y=-size / 2., z=wall_width / 2.))
    wall3 = create_box(wall_width, size + wall_width, wall_width, color=GREY)
    set_point(wall3, Point(x=size / 2., z=wall_width / 2.))
    wall4 = create_box(wall_width, size + wall_width, wall_width, color=GREY)
    set_point(wall4, Point(x=-size / 2., z=wall_width / 2.))

    walls = [wall1, wall2, wall3, wall4]
    #print(f"WALLS: {walls}")
    return walls


def create_obstacles(obstacle_list):
    '''
    input: list of obstacle locations
    Sets the obstacles in our pybullet environment
    output: List of obstacle ids.
    '''
    obstacle_ids = []
    for size, position in obstacle_list:
        w, l, h = size
        obstacle = create_box(w=w, l=l, h=h, color=GREY)  # Creates a box as an obstacle
        set_point(obstacle, position)  # Sets the [x,y,z] position of the obstacle
        # print('Position:', get_point(obstacle))

        # yaw = np.random.choice([np.pi / 4, np.pi / 3, np.pi / 2, np.pi])
        yaw = np.pi / 2
        set_euler(obstacle, [0, 0, yaw])  # Sets the [roll,pitch,yaw] orientation of the obstacle
        # print('Orientation:', get_euler(obstacle))
        obstacle_ids.append(obstacle)
        # print(f"OBSTACLE IDS: {obstacle}")
    return obstacle_ids


def add_robots(robots):
    '''
    Input: List of tuples containing  position, orientation, color, and goal.
    returns:
        robot_list: A list of dictionaries containing the name of the robot, start position, goal position, and sizes
    '''
    robot_list = []
    MODEL_DIRECTORY = os.path.join(os.path.abspath(''), 'turtlebot_models/turtlebot')
    TURTLEBOT_URDF = os.path.join(MODEL_DIRECTORY, 'turtlebot_holonomic.urdf')
    for position, orientation, color, goal in robots:
        orientation = p.getQuaternionFromEuler(orientation)
        robot = p.loadURDF(TURTLEBOT_URDF, position, orientation, useFixedBase=True,
                           flags=p.URDF_MERGE_FIXED_LINKS, globalScaling=1.5)
        change_color(robot, color)
        goal_marker = create_sphere(0.1, color=color)
        set_point(goal_marker, goal)

        rob_pos, rob_orient = p.getBasePositionAndOrientation(robot)
        # print(rob_pos, rob_orient)
        aabb = p.getAABB(robot)
        sizes = np.array(aabb[1]) - np.array(aabb[0])
        robot_info = {
            "name": robot,
            "start": position,
            "goal": goal,
            "sizes": sizes
        }
        robot_list.append(robot_info)

    return robot_list


def translate_path(path, robot_list, map_size):
    """
    Convert path array returned by primal to timestep dict
    :param path: path array returned by primal or filename to load path from .npy file
    :param robot_list: list of robot ids
    :return: path dictionary in this form:
    {
        robot_id: [{'t': .., 'x': x, 'y': y}, ...],
        ...
    }
    """
    if path is None:
        return None
    if isinstance(path, str):
        path = np.load(path)
    t = 0
    path_dict = {rob["name"]: [] for rob in robot_list}
    for robot_pos in path:
        for i, (row, col) in enumerate(robot_pos):
            robot_id = robot_list[i]["name"]
            path_dict[robot_id].append({
                't': t,
                'x': col - int(map_size / 2),
                'y': -row + int(map_size / 2)
            })
        t += 1
    return path_dict


def get_states(world):

    '''
    Input:
        world: 3d array containing a start matrix and goal matrix. 
        Start matrix possess start location and obstacles.
        End location posses goal locations for each agent.
    
    Returns: 
        start states, goal states, and the list of obstacles
    '''
    start_map = world[0]  # possesses start location and obstacles!
    world_size = len(start_map)
    goal_map = world[1]  # my understanding of what the second grid represents!
    start_states = {}
    obstacle_list = []
    goal_states = {}
    for row in range(len(start_map)):
        for col in range(len(start_map[row])):
            if int(start_map[row][col]) > 0:  # an agent
                agent = start_map[row][col]
                start_states[agent] = (col - int(world_size / 2), -row + int(world_size / 2), 0)
            elif int(start_map[row][col]) < 0:  # obstacle
                obstacle_list.append((col - int(world_size / 2), -row + int(world_size / 2), 0))  # x,y, orientation

    for row in range(len(goal_map)):
        for col in range(len(goal_map[row])):
            if int(goal_map[row][col]) > 0:  # agent end state
                agent = goal_map[row][col]
                goal_states[agent] = (col - int(world_size / 2), -row + int(world_size / 2), 0)

    return start_states, goal_states, obstacle_list


def setup_env(n_robots, size, start_states, goal_states, obstacle_states, disable_gui=False):
    '''

    Key function for setting up the pybullet simulation.

    Input: 
        n_robots is the number of robots in environment
        size is the size of the world
        start_states are the start states of the robots
        goal_states are the goal states of the robots
        obstacle_states are the locations of the obstacles
        disable_gui a flag to either present the user with a gui of our simulation or without

    returns:
        robot_list: List of robots containing their information about name, start position, goal position, and size
        ROBOTS: List of robots containing their information about start position, orientation, color, and goal
        OBSTACLE LIST: List of obstacle locations for our environment

    '''
    plane_id = p.loadURDF("plane.urdf")

    creat_walls(size=size + 2)
    map_dict = {}
    obs_pos = obstacle_states
    OBSTACLE_LIST = []
    for pos in obs_pos:
        OBSTACLE_LIST.append([(.999, .999, 1), pos])

    rob_start_pos = start_states
    rob_goal_pos = goal_states
    colors = np.random.choice(range(256), size=(n_robots, 3)) / 256
    ones = np.ones(n_robots)
    colors = np.column_stack((colors, ones))
    ROBOTS = []
    for i in range(1, n_robots + 1):
        r, g, b, a = colors[i - 1]
        color = RGBA(r, g, b, a)
        ROBOTS.append([rob_start_pos[i], [0, 0, 0], color, rob_goal_pos[i]])

    if disable_gui:
        robot_list = []
        for i, (position, orientation, color, goal) in enumerate(ROBOTS):
            robot_list.append({
                "name": i,
                "start": position,
                "goal": goal,
                "sizes": np.array([0.5, 0.5, 0.5])
            })
    else:
        obstacle_ids = create_obstacles(obstacle_list=OBSTACLE_LIST)
        robot_list = add_robots(ROBOTS)
    # print(robot_list)
    print('Env setup done...')
    return robot_list, ROBOTS, OBSTACLE_LIST


def to_2d(robot_list, OBSTACLE_LIST, sizeMap):
    """
    Convert robot configurations from 3D to 2D for CBS algorithm
    :param robot_list: list of robots with configurations
    :return: 2D info to feed in CBS
    """
    robot_list_2d = []
    for robot in robot_list:
        rob = {
            "name": robot["name"],
            "start": robot["start"][:2],
            "goal": robot["goal"][:2]
        }
        robot_list_2d.append(rob)

    obstacle_list_2d = []
    for size, pos in OBSTACLE_LIST:
        obstacle_list_2d.append([size[:2], pos[:2]])
    map_sizes = [sizeMap + 2, sizeMap + 2]
    robot_sizes = robot_list[0]["sizes"]
    robot_sizes_2d = robot_sizes[:2]

    return map_sizes, robot_list_2d, obstacle_list_2d, robot_sizes_2d


def planning(robot_list, OBSTACLE_LIST, time_constraint, mapSize):
    '''
    Function for finding a path using the CBS algorithm 
    Input:
        robot_list: List of robots with configurations
        OBSTACLE_LIST: List of obstacles
        time_constraint: Time allotted to find a solution
        mapSize = size of the map
    
    Output:
        path: Path found by the CBS algorithm
        duration: Time  in seconds to find the solution

        If could not find path, returns None
    '''
    # convert to 2D
    dimension, agents, obstacles, agent_sizes = to_2d(robot_list, OBSTACLE_LIST, mapSize)
    if not verify_arguments(dimension, agents, obstacles):
        return None

    path, duration = cbs.find_path(dimension, agents, obstacles, agent_sizes, time_constraint)
    if path:
        print("Found path in: {:.2f}s".format(duration))
    # print("Path found")
    return path, duration


def verify_arguments(dimension, agents, obstacles):
    """
    Function to make sure none of the start or goal states are outside the boundary of the map nor
    are any of the obstacles
    :param dimension: tuple containing x value, y value
    :param agents: number of agents
    :param obstacles: number of obstacles
    :return:
        True if the start and/or goal states are within the boundary. False otherwise
    """
    bounds = [(0 - dimension[0] / 2 + 1, 0 + dimension[0] / 2 - 1),
              (0 - dimension[1] / 2 + 1, 0 + dimension[1] / 2 - 1)]
    start_states = [tuple(agent['start']) for agent in agents]
    goal_states = [tuple(agent['goal']) for agent in agents]
    if len(start_states) > len(set(start_states)) or len(goal_states) > len(set(goal_states)):
        print("Your robots may share the same start or goal states.")
        return False
    for agent in agents:
        start_state = agent['start']
        goal_state = agent['goal']
        # checks that no agent starts or ends outside our map boundary
        for bound, start_dim, goal_dim in zip(bounds, start_state, goal_state):
            if bound[0] <= start_dim <= bound[1] and bound[0] <= goal_dim <= bound[1]:
                continue  # if neither start nor goal are on bounds or outside the bounds of our map
            else:
                return False
        # make sure that no start states start on an obstacle or goal states end on an obstacle
        for obstacle in obstacles:
            obs_center = obstacle[1]
            if start_state == obs_center or goal_state == obs_center:
                # print(f"AGENT {agent['name']} starts or ends on an obstacle")
                return False
    return True


def follow_path(path, num_steps=20):
    '''
    Function used to follow a path for CBS algorithm
    input:
        path: Path found by CBS algorithm
        num_steps: used to discretize our step size
    '''
    if path is None:
        print("There is no path to follow.")
        return
    robot_ids = list(path.keys())
    base_joints = joints_from_names(robot_ids[0], BASE_JOINTS)
    time_step = 0
    max_time_step = 0
    for _, positions in path.items():
        max_time_step = max(max_time_step, len(positions) - 1)

    simulation = {}
    while time_step <= max_time_step:
        for robot_id, positions in path.items():
            if time_step >= len(positions) - 1:
                simulation.pop(robot_id, None)
                continue
            current_pos = np.array([positions[time_step]["x"], positions[time_step]["y"]])
            next_pos = np.array([positions[time_step + 1]["x"], positions[time_step + 1]["y"]])
            unit_step = (next_pos - current_pos) / num_steps
            # print("Step: ", robot_id, current_pos, next_pos)
            unit_step_3d = np.append(unit_step, 0)
            simulation[robot_id] = unit_step_3d

        for s in range(num_steps):
            for robot_id, step in simulation.items():
                cur_pos, cur_orient = p.getBasePositionAndOrientation(robot_id)
                cur_pos = np.array(cur_pos) + step
                p.resetBasePositionAndOrientation(robot_id, cur_pos, cur_orient)
                # current_pos = current_pos + unit_step
                # set_joint_positions(robot_id, base_joints, current_pos)
            wait_for_duration(1 / num_steps)

        time_step += 1
    print("Simulation done.")


def run_offline():
    # num_agents, size, density, id = 8, 20,  .1, 37
    '''
        Function used to simulate PRIMAL
    '''
    num_agents, size, density, id = 4, 40, 0, 2
    environment_path = "PRIMAL/saved_environments"
    environment_data_filename = make_name(num_agents, size, density, id, ".npy", environment_path,
                                          extra="environment")
    world = np.load(environment_data_filename)
    start_states, goal_states, obstacle_states = get_states(world)  # gets location in grid, need to convert
    robot_list, _, OBSTACLE_LIST = setup_env(num_agents, size, start_states, goal_states, obstacle_states)
    results_path = './PRIMAL/primal_results'
    solution_filename = make_name(num_agents, size, density, id, ".npy", results_path, extra="solution")
    path = translate_path(solution_filename, robot_list, size)
    follow_path(path)


def simulate(args):

    '''
    Function used to for testing the success rate of PRIMAL and CBS.

    Input:
        n_tests: number of tests to run
        alg: algorithm to test
    '''
    # first terminal is 10 by 10
    # second terminal is 20 by 20
    # third terminal is 40 by 40
    n_tests = args.n_tests
    alg = args.alg

    env_tuples = []
    num_agents = 2
    suc_rate_list = []
    duration_list = []

    primal = PRIMAL('./PRIMAL/model_primal_1', 10)

    size = 40
    print(f"RUNNING PRIMAL W/ SIZE: {size}")
    density = 0.1
    agents = []
    rng = np.random.RandomState(2810)
    while num_agents < 32:
        num_agents *= 2
        # for size in [10, 20, 40, 80, 160]:
        if size == 10 and num_agents > 32: continue
        if size == 20 and num_agents > 128: continue
        if size == 40 and num_agents > 512: continue
        agents.append(num_agents)
        n_success = 0
        durations = []

        # for id in range(n_tests):
        for id in rng.choice(100, n_tests):
            env_tuples.append((num_agents, size, density, id))
            # num_agents, size, density, id = 8, 20, 0.3, 80
            # num_agents, size, density, id = 8, 20, 0.1, 37
            print("Starting tests for env: ", num_agents, size, density, id)

            environment_path = "PRIMAL/saved_environments"
            environment_data_filename = make_name(num_agents, size, density, id, ".npy", environment_path,
                                                  extra="environment")
            world = np.load(environment_data_filename)
            start_states, goal_states, obstacle_states = get_states(
                world)  # gets location in grid, need to convert
            robot_list, _, OBSTACLE_LIST = setup_env(num_agents, size, start_states, goal_states,
                                                         obstacle_states, disable_gui=args.no_gui)
            path = None
            duration = 0

            if args.alg == "cbs":
                path, duration = planning(robot_list, OBSTACLE_LIST, args.time_constraint, size)
            elif args.alg == "primal":
                results, path = run_simulations((num_agents, size, density, id), primal)
                duration = results['time']

            if path is not None:
                print('Path found')
                n_success += 1
                durations.append(duration)

                if not args.no_gui:
                    if alg == "primal":
                        path = translate_path(path, robot_list, size)
                    follow_path(path)
                    p.resetSimulation()
        success_rate = n_success / n_tests
        mean = np.mean(durations) if len(durations) else 0
        std = np.std(durations) if len(durations) else 0
        suc_rate_list.append(success_rate)
        duration_list.append(mean)
        print('Success rate: {:.2f}, mean/stev duration: {:.2f} / {:.2f}'.format(success_rate, mean, std))

    if not os.path.isdir('./stats'):
        os.makedirs('./stats')
    filename = './stats/{}_suc_rate_size{}_den{}_ntests{}.npy'.format(alg, size, density, n_tests)
    print("suc_rate_list", suc_rate_list)
    np.save(filename, suc_rate_list)
    filename = './stats/{}_duration_size{}_den{}_ntests{}.npy'.format(alg, size, density, n_tests)
    print("duration_list", duration_list)
    np.save(filename, duration_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-gui", action='store_true')
    parser.add_argument("--offline", action='store_true', help="Run simulation offline.")
    parser.add_argument("--n-tests", type=int, default=1, help="Number of tests to run")
    parser.add_argument("--time-constraint", type=int, default=60, help="Time limit (sec) for a solution to be found")
    parser.add_argument("--alg", type=str, default='cbs', choices=['cbs', 'primal'],
                        help="Type of algorithm to use. Options: cbs or primal")

    args = parser.parse_args()

    # or p.DIRECT for non-graphical version
    physicsClient = p.connect(p.DIRECT if args.no_gui else p.GUI, options='--mp4=demo.mp4 --fps=120')
    set_preview(False)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
    # Set gravity for simulation
    p.setGravity(0, 0, -9.8)
    # Set camera angles
    p.resetDebugVisualizerCamera(20, 15, -30, [.0, -0.0, -0.0])

    print("Args: ", args)

    if args.offline:
        run_offline()
    else:
        simulate(args)

    # while 1:
    #     p.stepSimulation()
    #     time.sleep(1. / 240)
    p.disconnect()


if __name__ == '__main__':
    main()
