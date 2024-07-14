import sys
sys.path.append('')

from odin.actions import action_map
from odin.utils.default import default_arguments_and_behavior

if __name__ == "__main__":
    kwargs = default_arguments_and_behavior()

    action_map[kwargs['action']](**kwargs)
