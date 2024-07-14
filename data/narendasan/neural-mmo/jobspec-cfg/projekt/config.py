from pdb import set_trace as TT
import numpy as np
from forge.blade import core
import os

def get_experiment_name(config):
   #  assert len(config.SKILLS) == 1
   experiment_name = 'fit-{}_skills-{}_gene-{}_algo-{}'.format(
      config.FITNESS_METRIC,
      config.SKILLS,
      config.GENOME,
      config.EVO_ALGO,
   )

   if config.PAIRED:
      experiment_name += '_PAIRED'

   if config.EVO_ALGO == 'MAP-Elites':
      #     experiment_name += '_BCs-{}'.format(config.ME_DIMS)
      if (np.array(config.ME_BIN_SIZES) == 1).all():
         experiment_name += '_noBCs'

   if config.SINGLE_SPAWN:
      experiment_name += '_uniSpawn'
   experiment_name += '_' + config.EVO_DIR

   return experiment_name

class Base(core.Config):
   '''Base config for RLlib Models
   ENV_NAME                = 'Neural_MMO'
   ENV_VERSION             = '1.5'
   NUM_WORKERS             = 6
   NUM_GPUS_PER_WORKER     = 0
   NUM_GPUS                = 1
   TRAIN_BATCH_SIZE        = 4800 # to match evo, normally 4000
   #TRAIN_BATCH_SIZE        = 400
   Extends core Config, which contains environment, evaluation,
   and non-RLlib-specific learning parameters'''

   MELEE_MULT = 63 / 99
   RANGE_MULT = 32 / 99
   MAGE_MULT =  24 / 99
   #Hardware Scale
   NUM_WORKERS             = 4
   NUM_GPUS_PER_WORKER     = 0
   NUM_GPUS                = 1
   LOCAL_MODE              = False

   #Memory/Batch Scale
   TRAIN_BATCH_SIZE        = 400000
   ROLLOUT_FRAGMENT_LENGTH = 100

   #Optimization Scale
   SGD_MINIBATCH_SIZE      = 128
   NUM_SGD_ITER            = 1

   #Model Parameters 
   #large-map:        Large maps baseline
   #small-map:        Small maps baseline
   #scripted-combat:  Scripted with combat
   #scripted-forage:  Scripted without combat
   #current:          Resume latest checkpoint
   #None:             Train from scratch
   MODEL                   = 'current'
   N_AGENT_OBS             = 100
   NPOLICIES               = 1
   HIDDEN                  = 64
   EMBED                   = 64

   #Scripted model parameters
   SCRIPTED_BACKEND        = 'dijkstra' #Or 'dynamic_programming'
   SCRIPTED_EXPLORE        = True       #Intentional exploration


class LargeMaps(Base):
   '''Large scale Neural MMO training setting

   Features up to 1000 concurrent agents and 1000 concurrent NPCs,
   1km x 1km maps, and 5/10k timestep train/eval horizons

   This is the default setting as of v1.5 and allows for large
   scale multiagent research even on relatively modest hardware'''

   NAME                    = __qualname__
   MODEL                   = 'large-map'

   PATH_MAPS               = core.Config.PATH_MAPS_LARGE

   TRAIN_HORIZON           = 5000
   EVALUATION_HORIZON      = 10000

   NENT                    = 1024
   NMOB                    = 1024


class SmallMaps(Base):
   '''Small scale Neural MMO training setting

   Features up to 128 concurrent agents and 32 concurrent NPCs,
   60x60 maps (excluding the border), and 1000 timestep train/eval horizons.
   
   This setting is modeled off of v1.1-v1.4 It is appropriate as a quick train
   task for new ideas, a transfer target for agents trained on large maps,
   or as a primary research target for PCG methods.'''

   NAME                    = __qualname__
   MODEL                   = 'small-map'
   SCRIPTED_EXPLORE        = False

   TRAIN_HORIZON           = 1000
   EVALUATION_HORIZON      = 1000

   NENT                    = 128
   NMOB                    = 0

   #Path settings
   PATH_MAPS               = core.Config.PATH_MAPS_SMALL
   PATH_ROOT               = os.path.join(os.getcwd(), PATH_MAPS, 'map')

   #Outside-in map design
   SPAWN_CENTER            = False
   INVERT_WILDERNESS       = True
   WILDERNESS              = False

   #Terrain generation parameters
   TERRAIN_MODE            = 'contract'
   TERRAIN_LERP            = False
   TERRAIN_SIZE            = 80 
   TERRAIN_OCTAVES         = 1
   TERRAIN_FOREST_LOW      = 0.30
   TERRAIN_FOREST_HIGH     = 0.75
   TERRAIN_GRASS           = 0.715
   TERRAIN_ALPHA           = -0.025
   TERRAIN_BETA            = 0.035

   #Entity spawning parameters
   PLAYER_SPAWN_ATTEMPTS   = 1
   NPC_LEVEL_MAX           = 35
   NPC_LEVEL_SPREAD        = 5
   NPC_SPAWN_PASSIVE       = 0.00
   NPC_SPAWN_NEUTRAL       = 0.60
   NPC_SPAWN_AGGRESSIVE    = 0.80


#<<<<<<< HEAD
ALL_SKILLS = ['constitution', 'fishing', 'hunting', 'range', 'mage', 'melee', 'defense', 'woodcutting', 'mining', 'exploration',]
COMBAT_SKILLS = ['range', 'mage', 'melee']
EXPLORE_SKILLS = ['exploration']
HARVEST_SKILLS = ['woodcutting', 'mining']

class TreeOrerock(SmallMaps):
   load_arguments = -1  # load a settings json with a particular index (or not if -1)
   NMAPS = SmallMaps.N_TRAIN_MAPS
   NTILE = 9
   NEW_EVAL = False
   EVO_MAP = True
   FIXED_MAPS = True
   EVALUATE = True
   NENT                 = 16
   NMOB                 = 0
   MODEL                = 'current'
   TERRAIN_SIZE         = 70
#   TERRAIN_DIR          = Base.TERRAIN_DIR_SMALL
   ROOT                 = os.path.join(os.getcwd(), Base.PATH_MAPS_SMALL, 'map')
#  TERRAIN_RENDER       = True
#  TERRAIN_ALPHA = 0
#  TERRAIN_BETA = 0
#  TERRAIN_WATER        = 0.25
#  TERRAIN_FOREST_LOW   = 0.35
   TERRAIN_GRASS_0   = 0.4
   TERRAIN_LAVA  = 0.45
   TERRAIN_SPAWN = 0.5
#  TERRAIN_GRASS        = 0.7
#  TERRAIN_FOREST_HIGH  = 0.725
   TERRAIN_TREE         = 0.8
   TERRAIN_OREROCK      = 0.85
   GRIDDLY = False
#  SKILLS               = 'ALL'
   SKILLS               = ALL_SKILLS
   FITNESS_METRIC       = 'L2'
   MAP = 'PCG'
   INFER_IDX = 0
   N_EVAL = 20
   EVO_VERBOSE          = True
   EVO_SAVE_INTERVAL    = 5
   GRIDDLY = False
   EVO_DIR = None
   PRETRAIN = False
   # reward agents for collective skill-diversity rather than the usual survival reward?
   REWARD_DIVERSITY = False
   PAIRED = False
   VIS_MAPS = False
   MULTI_MODEL_EXPERIMENTS = None
   MULTI_MODEL_NAMES = None
   MAX_POP = None  # Maximum number of agents belonging to a given population.
   GENOME = 'Baseline'


ALL_SKILLS = ['constitution', 'fishing', 'hunting', 'range', 'mage', 'melee', 'defense', 'woodcutting', 'mining', 'exploration',]
COMBAT_SKILLS = ['range', 'mage', 'melee']
EXPLORE_SKILLS = ['exploration']
HARVEST_SKILLS = ['woodcutting', 'mining']

class EvoNMMO(TreeOrerock):
   N_GENERATIONS = 10000
   FIXED_MAPS = False
   EVALUATE = False
 # INFER_IDX = 79766
 # INFER_IDX = 80117
   # How to measure diversity of agents on generated map.
   FITNESS_METRIC = 'L2' # 'Differential', 'L2', 'Discrete', 'Hull', 'Sum', 'Lifespans', 'Actions'
   GENOME = 'TileFlip'  # CPPN, Primitives, TileFlip...
   THRESHOLD = False
   TERRAIN_MODE = 'contract'
   EVO_MAP = True
   RENDER = False
   MODEL = 'current'
   NENT = 16  # Maximum population size
   TERRAIN_SIZE = 70
   EVO_DIR = '0'
   ROOT = os.path.join(os.getcwd(), 'evo_experiment', EVO_DIR, 'maps', 'map')
   N_EVO_MAPS = 48
   MAX_STEPS = 100
   MATURE_AGE = 3
   ROLLING_FITNESS = 25  # Size of window to use while calculating mean rolling fitness in cases where elite maps are re-evaluated after insertion into archive
   TERRAIN_RENDER = False
   TERRAIN_WATER        = 0.15
   TERRAIN_GRASS        = 0.35
   TERRAIN_LAVA         = 0.45
   TERRAIN_FOREST_LOW   = 0.55
   TERRAIN_FOREST_HIGH  = 0.7
   TERRAIN_TREE         = 0.8
   TERRAIN_OREROCK      = 0.9
   NET_RENDER = False
#  SKILLS = ['exploration']
#  SKILLS = ['woodcutting', 'mining']
#  SKILLS = ['range', 'mage', 'melee']
   EVO_ALGO = 'Simple'  # Simple, MAP-Elites, NEAT
   N_PROC = 6
   PRETRAINED = False
#  MAP_DIMS = ['woodcutting', 'mining']
   ME_DIMS = ['mining', 'woodcutting']
   ME_BIN_SIZES = [50, 50]
   ME_BOUNDS = [(0, 100), (0, 100)]
   ARCHIVE_UPDATE_WINDOW = 0  # How long without any updates to ME archive before randomly re-evaluating some elites?
   FEATURE_CALC = 'map_entropy'
   TEST = False
   ITEMS_PER_BIN = 1
   FROZEN = False
   SINGLE_SPAWN = False
   SKILLS = 'ALL'

class Explore(EvoNMMO):
   SKILLS = EXPLORE_SKILLS

class Combat(EvoNMMO):
   SKILLS = COMBAT_SKILLS

class All(EvoNMMO):
   SKILLS = ALL_SKILLS

class Griddly(EvoNMMO):
   TRAIN_RENDER = False
   GRIDDLY = True
   REGISTERED = False  #FIXME: hack. Do not set this.
   TEST = False
   NENT=5
   PRETRAIN = False
   EVO_DIR = 'griddly_scratch_0'
   TERRAIN_BORDER = 1
   ME_DIMS = ['mine_skill', 'woodcut_skill']
   ME_BIN_SIZES = [100, 100]
   ME_BOUNDS = [(0, 50), (0, 50)]
   SKILLS = ['drink_skill', 'gather_skill', 'woodcut_skill', 'mine_skill']

class Debug(SmallMaps):
   '''Debug Neural MMO training setting

   A version of the SmallMap setting with greatly reduced batch parameters.
   Only intended as a tool for identifying bugs in the model or environment'''
   MODEL                   = None
   LOCAL_MODE              = True
   NUM_WORKERS             = 1

   TRAIN_BATCH_SIZE        = 400
   TRAIN_HORIZON           = 200
   EVALUATION_HORIZON      = 50

   HIDDEN                  = 2
   EMBED                   = 2

