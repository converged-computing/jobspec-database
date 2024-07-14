#!/usr/bin/env python

# Usage
# python3 pdf.py 
#   or
# python3 pdf.py HEADER DATA FACTORS LOGZ JSON_IN NEX_OUT

import json
import numpy as np
import pandas as pd
import sys, getopt

if len(sys.argv) != 7:
    header_file = "CombineDS_header.csv"
    data_file = "CombineDS.csv"
    # Note: factors at the end of branches!
    factors_file = "factors_end.csv"
    logZ_file = "logz.txt"
    tree_phyjson_file = "tre.phyjson"
    output_file = "output.nex"
else:
    header_file = sys.argv[1]
    data_file = sys.argv[2]
    factors_file = sys.argv[3]
    logZ_file = sys.argv[4]
    tree_phyjson_file = sys.argv[5]
    output_file = sys.argv[6]

###

columns = pd.read_csv(header_file, delimiter=', ', engine='python').columns
data = pd.read_csv(data_file, header=None, delimiter=', ', engine='python')
logZ = pd.read_csv(logZ_file, header=None)
factors = pd.read_csv(factors_file, header=None, delimiter=', ', engine='python')
with open(tree_phyjson_file) as f:
    tree = json.load(f)

data.columns = columns
data['logZ'] = logZ[0].repeat(len(data) / len(logZ)).values
data['W'] = np.exp(data['logZ'] - data['logZ'].max())
data['W'] /= data['W'].sum()

root = tree['trees'][0]['root']
taxa = {t['id']: t['name'] for t in tree['taxa']}

lambda0 = data['W'] * data['lambda_0.k'] * data['lambda_0.theta']
mu0 = data['W'] * data['mu_0.k'] * data['mu_0.theta']

def process(node, idx=0):
    if idx > 0:
        node['lambda'] = np.sum(lambda0 * factors[idx])
        node['mu'] = np.sum(mu0 * factors[idx])
    else:
        node['lambda'] = np.sum(lambda0)
        node['mu'] = np.sum(mu0)
    node['div'] = node['lambda'] - node['mu']
    idx += 1
    if 'children' in node:
        for child in node['children']:
            idx = process(child, idx)
    return idx

process(root)

def newick(node):
    res = ""
    if node.get('children'):
        res += "(" + ",".join(newick(child) for child in node['children']) + ")"
    if 'taxon' in node:
        res += f"'{taxa[node['taxon']]}'"
    if node['branch_length'] >= 0.0:
        res += f":{node['branch_length']}"
    res += f"[&Lambda={node['lambda']},Mu={node['mu']},Div={node['div']}]"
    return res

with open(output_file, "w") as output:
    output.write("""\
#NEXUS
begin trees
    tree t =  """)
    output.write(newick(root) + ";")
    output.write("""
end;

begin figtree;
    set appearance.backgroundColorAttribute="Default";
    set appearance.backgroundColour=#ffffff;
    set appearance.branchColorAttribute="Div";
    set appearance.branchColorGradient=true;
    set appearance.branchLineWidth=2.0;
    set appearance.branchMinLineWidth=0.0;
    set appearance.branchWidthAttribute="Fixed";
    set appearance.foregroundColour=#000000;
    set appearance.hilightingGradient=false;
    set appearance.selectionColour=#2d3680;
    set branchLabels.colorAttribute="User selection";
    set branchLabels.displayAttribute="Branch times";
    set branchLabels.fontName="sansserif";
    set branchLabels.fontSize=8;
    set branchLabels.fontStyle=0;
    set branchLabels.isShown=false;
    set branchLabels.significantDigits=4;
    set layout.expansion=0;
    set layout.layoutType="RECTILINEAR";
    set layout.zoom=0;
    set legend.attribute="Lambda";
    set legend.fontSize=10.0;
    set legend.isShown=true;
    set legend.significantDigits=4;
    set nodeBars.barWidth=4.0;
    set nodeBars.displayAttribute=null;
    set nodeBars.isShown=false;
    set nodeLabels.colorAttribute="User selection";
    set nodeLabels.displayAttribute="Node ages";
    set nodeLabels.fontName="sansserif";
    set nodeLabels.fontSize=8;
    set nodeLabels.fontStyle=0;
    set nodeLabels.isShown=false;
    set nodeLabels.significantDigits=4;
    set nodeShapeExternal.colourAttribute="User selection";
    set nodeShapeExternal.isShown=false;
    set nodeShapeExternal.minSize=10.0;
    set nodeShapeExternal.scaleType=Width;
    set nodeShapeExternal.shapeType=Circle;
    set nodeShapeExternal.size=4.0;
    set nodeShapeExternal.sizeAttribute="Fixed";
    set nodeShapeInternal.colourAttribute="User selection";
    set nodeShapeInternal.isShown=false;
    set nodeShapeInternal.minSize=10.0;
    set nodeShapeInternal.scaleType=Width;
    set nodeShapeInternal.shapeType=Circle;
    set nodeShapeInternal.size=4.0;
    set nodeShapeInternal.sizeAttribute="Fixed";
    set polarLayout.alignTipLabels=false;
    set polarLayout.angularRange=0;
    set polarLayout.rootAngle=0;
    set polarLayout.rootLength=100;
    set polarLayout.showRoot=true;
    set radialLayout.spread=0.0;
    set rectilinearLayout.alignTipLabels=false;
    set rectilinearLayout.curvature=0;
    set rectilinearLayout.rootLength=100;
    set scale.offsetAge=0.0;
    set scale.rootAge=1.0;
    set scale.scaleFactor=1.0;
    set scale.scaleRoot=false;
    set scaleAxis.automaticScale=true;
    set scaleAxis.fontSize=8.0;
    set scaleAxis.isShown=false;
    set scaleAxis.lineWidth=1.0;
    set scaleAxis.majorTicks=1.0;
    set scaleAxis.minorTicks=0.5;
    set scaleAxis.origin=0.0;
    set scaleAxis.reverseAxis=false;
    set scaleAxis.showGrid=true;
    set scaleBar.automaticScale=true;
    set scaleBar.fontSize=10.0;
    set scaleBar.isShown=true;
    set scaleBar.lineWidth=1.0;
    set scaleBar.scaleRange=0.0;
    set tipLabels.colorAttribute="User selection";
    set tipLabels.displayAttribute="Names";
    set tipLabels.fontName="sansserif";
    set tipLabels.fontSize=8;
    set tipLabels.fontStyle=0;
    set tipLabels.isShown=true;
    set tipLabels.significantDigits=4;
    set trees.order=false;
    set trees.orderType="increasing";
    set trees.rooting=false;
    set trees.rootingType="User Selection";
    set trees.transform=false;
    set trees.transformType="cladogram";
end;
""")
