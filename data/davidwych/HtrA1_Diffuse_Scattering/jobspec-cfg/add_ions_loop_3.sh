#!/bin/bash

AddToBox -c 3num_loop_3_SC.pdb -a ./lithium/lithium.pdb -na 7510 -o 3num_loop_3_SC_Li.pdb -P 233640 -RP 3.0 -RW 6.0 -G 0.1 -V 1

AddToBox -c 3num_loop_3_SC_Li.pdb -a ./sodium/sodium.pdb -na 752 -o 3num_loop_3_SC_LiNa.pdb -P 241150 -RP 3.0 -RW 6.0 -G 0.1 -V 1

AddToBox -c 3num_loop_3_SC_LiNa.pdb -a ./sulfate/sulfate.pdb -na 5700 -o 3num_loop_3_SC_LiNaSO4.pdb -P 241902 -RP 3.0 -RW 6.0 -G 0.1 -V 1

AddToBox -c 3num_loop_3_SC_LiNaSO4.pdb -a ./ammonium/ammonium.pdb -na 3754 -o 3num_loop_3_SC_LiNaSO4NH4.pdb -P 247602 -RP 3.0 -RW 6.0 -G 0.1 -V 1

AddToBox -c 3num_loop_3_SC_LiNaSO4NH4.pdb -a ./citrate/citrate.pdb -na 380 -o 3num_loop_3_SC_LiNaSO4NH4Cit.pdb -P 251356 -RP 3.0 -RW 6.0 -G 0.1 -V 1
