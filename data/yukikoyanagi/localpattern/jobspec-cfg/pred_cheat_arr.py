#!/usr/bin/env python2
#
# Predict SO3 rotation from H-bond local pattern, but we 'cheat' by
# taking the n'th closest value to the true value. This allows us to
# investigate the magnitude of potential improvement, and the
# distribution of cluster modes.

MAXSTEP = 791

import argparse, os.path, re, glob, math, os
from itertools import islice
from operator import itemgetter

import Protein
import Option
import conv

def loadassess(ad):
    # Read assess file into dict
    adic = {}
    expr = (r'total observations: (?P<obs>\d+)'
            r'(.*\n)+?(?P<pat>^[^#]\S+)'
            r'\t\S+\t(?P<rot>\S+)'
            r'\t(?P<score>\d+.\d+)\s+')
    p = re.compile(expr, re.M)

    ad = ad.rstrip('/')
    for step in range(MAXSTEP+1):
        af = '{}/step{}_assess'.format(ad, step)
        with open(af) as fh:
            rtext = fh.read()
        it = p.finditer(rtext)
        for m in it:
            rot = map(float, re.split(r'[;,]', m.group('rot')))
            adic[(step, m.group('pat'))] = (rot,
                                            int(m.group('obs')),
                                            float(m.group('score')))
    return adic

def rotdif(p_rot, a_rot):
    x1, y1, z1, phi = p_rot
    n1 = math.sqrt(sum(map(lambda x: x**2, [x1, y1, z1])))
    x1, y1, z1 = x1/n1, y1/n1, z1/n1
    p_rot = conv.ev2mat(phi, [x1, y1, z1])
    psi, x2, y2, z2 = a_rot
    n2 = math.sqrt(sum(map(lambda x: x**2, [x2, y2, z2])))
    x2, y2, z2 = x2/n2, y2/n2, z2/n2
    a_rot = conv.ev2mat(psi, [x2, y2, z2])
    return conv.so3dist(p_rot, a_rot)
    

def predict(pf, ad, optd, od, nth):
    # Load protein
    pid = os.path.splitext(os.path.basename(pf))[0]
    prot = Protein.Protein(pid)
    prot.fromfiles(pf, None)

    # Load assess files into dict {(step, pat): (rot, #obs, score)}
    adic = loadassess(ad)

    optd = optd.rstrip('/')
    opts = []
    for step in range(MAXSTEP+1):
        of = '{}/step{}_opts'.format(optd, step)
        opt = Option.Option(of)
        opts.append(opt)

    results = []
    for bond in prot.hbonds:
        preds = []
        for step in range(MAXSTEP+1):
            pat = str(prot.findpattern2(bond, opts[step]))
            try:
                rot, _, score = adic[(step, pat)]
            except KeyError:
                continue
            preds.append((step, pat, score,
                          rotdif(bond.rotation, rot)))
        # We don't want ambiguous clusters
        preds = [pred for pred in preds if pred[2] > 10]
        preds.reverse()
        preds.sort(key=itemgetter(3))
        try:
            step, pat, score, dif = preds[nth-1]
        except IndexError:
            step, pat, score, dif = preds[-1]
        line = '{}\t{}\t{}\t{}\t{}\t{}'.format(pid, bond.lineno,
                                               step, pat, score, dif)
        if not od:
            print(line)
            continue

        results.append(line)

    fn = 'predch{}.{}.txt'.format(nth, pid)
    outf = os.path.join(od, fn)
    with open(outf, 'w') as fh:
        fh.write('\n'.join(results))


def main(pd, ad, optd, od, nth):
    if os.path.isfile(pd):
        predict(pd, ad, optd, od, nth)
    else:
        tcount = int(os.environ['SLURM_NTASKS'])
        tid = int(os.environ['SLURM_PROCID'])
        pd = pd.rstrip('/')
        plst = islice(glob.glob('{}/*.txt'.format(pd)),
                      tid, None, tcount)
        for pfile in plst:
            predict(pfile, ad, optd, od, nth)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pdir',
                        help='Directory containing protein files, '
                        'or a protein H-bond file. If directory '
                        'assume we are running under slurm.')
    parser.add_argument('assessd',
                        help='Directory containing *_assess files.')
    parser.add_argument('optd',
                        help='Directory containing _opts files.')
    parser.add_argument('-o', '--outd',
                        help='Output directory.')
    parser.add_argument('-n', '--nth', type=int, default=1,
                        help="Pick n'th best prediction instead of "
                        "the best.")
    args = parser.parse_args()
    main(args.pdir, args.assessd, args.optd, args.outd, args.nth)
