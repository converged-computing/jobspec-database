from ase.io import read
import sys


def write_kart_conf(atoms, name, fname):
    lines = []
    lines.append(f' run_id:            {name}\n')
    lines.append(f' total energy :    0.0000\n')
    cell = atoms.cell.lengths()
    lines.append(f'{cell[0]:10.4f}{cell[1]:10.4f}{cell[2]:10.4f}\n')
    sym2label={'Cu':1, 'O':2}
    for at in atoms:
        lines.append(f'{sym2label[at.symbol]:3d}{at.position[0]:12.6f}{at.position[1]:12.6f}{at.position[2]:12.6f}\n')

    with open(fname, 'w') as fhandle:
        for l in lines:
            fhandle.write(l)

if __name__ == '__main__':
    last_atoms = read('allconf', format='extxyz', index=-1)
    write_kart_conf(last_atoms, sys.argv[1], 'cont.conf')
