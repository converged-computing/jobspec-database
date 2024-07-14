import os
import sys
import re
import argparse

from delphin import itsdb
from delphin.itsdb import unescape
from delphin.derivation import Derivation, UdfTerminal, UdfToken

parser = argparse.ArgumentParser(description='Export out TSDB databases to TSV.')
parser.add_argument('-d', '--directory', type=str,
        help='The directory of the tsdb database.')
parser.add_argument('-o', '--output', type=str,
       help='The output name of the tsv.')
parser.add_argument('-n', '--includena', action='store_true',
        help='Include sentences without valid parses.')
parser.add_argument('-r', '--derivation', action='store_true',
        help='Include derivations.')
parser.add_argument('-p', '--preprocess', action='store_true',
        help='Include sentences with their unk\'ed entries.')
parser.add_argument('-e', '--parser-error', action='store_true',
        help='Include parser error messages.')
args = parser.parse_args()
print(args)

#STEP 1
def get_tokens(derivation):
    stack, trace, tokens = [ derivation ],  [ '' ], []

    while len(stack) != 0:
        node = stack.pop()
        cur = trace.pop()

        if type(node) == UdfTerminal:
            stack.extend(node.tokens)
            nex = cur + '/%s' % '/UdfTerminal'
            trace.extend([nex] * len(node.tokens))
        elif type(node) == UdfToken:
            tokens.insert(0, (node, cur))
        else:
            stack.extend(node.daughters)
            nex = cur + '/%s' % node.entity
            trace.extend([nex] * len(node.daughters))

    return tokens


#STEP 2
def preprocess(inp, derivation):
    derivation = Derivation.from_string(derivation)
    tokens = get_tokens(derivation)
    traces = [ i[1] for i in tokens ]
    
    sent = []
    for token, trace in tokens:
        lemma = trace.split('/')[-3]

        #native entry
        if not lemma.startswith('generic'):
            to = int(re.search(r'\+TO .*?\\"(\d+)\\"', 
                token.tfs).group(1))
            fro = int(re.search(r'\+FROM .*?\\"(\d+)\\"', 
                token.tfs).group(1))
            form = inp[fro:to]
        else:
            form = lemma

            #add punctuation
            if 'comma' in trace:
                form = '%s,' % form
            if 'asterisk_' in trace:
                form = '%s*' % form
            if 'asterisk-pre' in trace:
                form = '*%s' % form
            if 'threedot' in trace:
                form = '%s...' % form
            if 'hyphen' in trace:
                form = '%s-' % form
            if 'sqright' in trace:
                form = '%s\'' % form
            if 'sqleft' in trace:
                form = '\'%s' % form
            if 'dqright' in trace:
                form = '%s\'' % form
            if 'dqleft' in trace:
                form = '\'%s' % form
            if 'rparen' in trace:
                form = '%s)' % form
            if 'lparen' in trace:
                form = '(%s' % form
            if 'comma-rp' in trace:
                form = '%s,)' % form
            if 'bang' in trace:
                form = '%s!' % form
            if 'qmark' in trace:
                form = '%s?' % form
            if 'qmark-bang' in trace:
                form = '%s?!' % form
            if 'period' in trace:
                form = '%s.' % form

        #fix compounds
        if '-' in form and form[-1] != '-':
            form = form.split('-')[1]

        sent.append(form)
    return ' '.join(sent)


for i, fn in enumerate(os.listdir(args.directory)):
    output = open(args.output +
            '' if i == 0 else str(i), 'wt')

    prof = itsdb.TestSuite(os.path.join(args.directory, fn, 'pet/'))
    id_to_parse = { result['parse-id'] : result['derivation'] for result in prof['result'] }

    for item, parse in zip(prof['item'], prof['parse']):
        attr = []

        #input
        inp = item['i-input']
        attr.append(inp)

        #derivation
        deriv = id_to_parse.get(item['i-id'], 'NA')
        if args.derivation:
            attr.append(deriv.strip())
        
        #na
        if not args.includena:
            if deriv == 'NA':
                continue

        #preprocess unk string
        if deriv != 'NA':
            preproc = preprocess(inp, deriv)
            
            if args.preprocess:
                attr.append(preproc)

            if args.parser_error:
                attr.append('NA')

        else:
            if args.preprocess:
                attr.append('NA')

            if args.parser_error:
                attr.append(parse['error'] if parse['error'] != '' else 'NA')

        print(len(attr))
        output.write('\t'.join(attr) + '\n')
