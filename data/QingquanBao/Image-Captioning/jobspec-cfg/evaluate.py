import argparse
import pandas as pd
import numpy as np

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

from utils.util import ptb_tokenize

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_file", type=str)
    parser.add_argument("--reference_file", type=str)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()
    prediction_df = pd.read_json(args.prediction_file)
    #[n, var_len]
    key_to_pred = dict(zip(prediction_df["img_id"], prediction_df["prediction"]))
    #[n, 5, var_len]
    captions = open(args.reference_file, 'r').read().strip().split('\n')
    key_to_refs = {}
    for i, row in enumerate(captions):
        row = row.split('\t')
        row[0] = row[0][:len(row[0]) - 2]  # filename#0 caption
        if row[0] not in key_to_pred:
            continue
        if row[0] in key_to_refs:
            key_to_refs[row[0]].append(row[1])
        else:
            key_to_refs[row[0]] = [row[1]]

    scorers = [Bleu(n=4), Rouge(), Meteor(), Cider(), Spice()]
    key_to_refs = ptb_tokenize(key_to_refs)
    key_to_pred = ptb_tokenize(key_to_pred)

    all_res = np.empty((len(key_to_refs.keys()), 9))
    cur_col = 0

    output = {"SPIDEr": 0}
    with open(args.output_file, "w") as writer:
        for scorer in scorers:
            score, scores = scorer.compute_score(key_to_refs, key_to_pred)
            method = scorer.method()
            output[method] = score
            if method == "Bleu":
                for n in range(4):
                    print("Bleu-{}: {:.3f}".format(n + 1, score[n]), file=writer)
                    all_res[:, cur_col] = scores[n]
                    cur_col += 1
            else:
                print(f"{method}: {score:.3f}", file=writer)
                if method in ['SPICE']:
                    spice_score = [ ]
                    for img in scores:
                        img_score = np.array([x['f'] for x in img.values()])
                        spice_score.append(img_score[~np.isnan(img_score)].mean())
                    
                    all_res[:, cur_col] = spice_score
                    all_res[:, -1] += spice_score
                    cur_col += 1
                else:
                    all_res[:, cur_col] = scores
                    cur_col += 1
            if method in ["CIDEr", "SPICE"]:
                output["SPIDEr"] += score
                if method in ["CIDEr"]:
                    all_res[:, -1] = scores
        output["SPIDEr"] /= 2
        all_res[:, -1] /= 2
        print(f"SPIDEr: {output['SPIDEr']:.3f}", file=writer)

        all_csv = pd.DataFrame(all_res,
                                index=key_to_pred.keys(),
                                columns=['Bleu-1', 'Bleu-2','Bleu-3','Bleu-4',
                                         'Rouge', 'Meteor', 'Cider', 'Spice',
                                          'SPIDEr'])
        all_csv.to_csv('res.csv')
