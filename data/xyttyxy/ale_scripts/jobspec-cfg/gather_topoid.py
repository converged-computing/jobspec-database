import numpy as np
import os


class topo:
    def __init__(self, lines):
        line = lines[0].split()
        self.topoid = int(line[0])
        self.topo_labels = [int(l) for l in line[1:4]]
        if line[4] == 'T':
            self.topo_difficult = True
        else:
            self.topo_difficult = False

        self.cutoffs = np.zeros((2,2))
        line = lines[1].split()
        self.cutoffs[0,0] = float(line[0])
        self.cutoffs[0,1] = float(line[1])
        line = lines[2].split()
        self.cutoffs[1,0] = float(line[0])
        self.cutoffs[1,1] = float(line[1])

        line = lines[3].split()
        self.count = int(line[0])
        self.attempts = int(line[1])
        self.failed_attempts = int(line[2])
        self.atoms_in_cluster = int(line[3])
        self.primary_topoid = int(line[4])

    def __repr__(self):
        line0 = f'{self.topoid}'.rjust(14)
        line0 += ''.join([f'{elm}'.rjust(14) for elm in self.topo_labels])
        line0 += ' F\n'

        line1 = f'{self.cutoffs[0,0]:10.4f}{self.cutoffs[0,1]:10.4f}\n'
        line2 = f'{self.cutoffs[1,0]:10.4f}{self.cutoffs[1,1]:10.4f}\n'

        line3 = ''.join([f'{elm}'.rjust(14) for elm in [self.count, self.attempts, self.failed_attempts, self.atoms_in_cluster, self.primary_topoid]])+'\n'
        return ''.join([line0, line1, line2, line3])

    def __str__(self):
        names = ['topoid', 'count', 'attempts', 'failed', 'primary_topoid']
        values = [self.topoid, self.count, self.attempts, self.failed_attempts, self.primary_topoid]
        l1 = ''.join([f'{n:15s}' for n in names])+'\n'
        l2 = ''.join([f'{n:15d}' for n in values])+'\n'
        return l1 + l2

def read_toposlist(fname):
    with open(fname, 'r')  as topolist:
        all_lines = topolist.readlines()[3:]
    numlines = len(all_lines)
    numtopos = (numlines)/4

    assert numtopos.is_integer(), 'number of topos does not make sense'
    numtopos = int(numtopos)
    topos = []
    for t in range(numtopos):
        start = t*4
        end = (t+1)*4
        topos.append(topo(all_lines[start:end]))
    return topos

if __name__ == '__main__':
    topos = read_toposlist('topos.list')
    searches = sorted([d for d in os.listdir('.') if 'search_0' in d])
    for s in searches:
        fname = f'{s}/topos.list'
        new_topos = read_toposlist(fname)
        insert = True
        for nt in new_topos:
            for it, tt in enumerate(topos):
                if nt.topoid == tt.topoid:
                    # it is not new but there are searches not previously accounted for
                    insert = False
                    topos[it].attempts += nt.attempts
                    topos[it].failed_attempts += nt.failed_attempts

            # if this topology is not found in the existing list of topologies
            # it is new and should be added to the list
            if insert:
                topos.append(nt)
    
    # keep in mind if you execute this over and over again the search counts are just gonna keep increasing
    with open('topos.list.gathered', 'w') as topolist:
        for t in topos:
            for l in t.__repr__():
                topolist.write(l)
    

