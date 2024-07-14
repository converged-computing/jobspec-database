import src
from sklearn.model_selection import train_test_split
import pandas as pd


features = {
    "tfidf": {},
    "fasttext": {}
}
options = {
    "unique": True,
    "only_wrong": False,
    "min_wordcount": {
        "enabled": True,
        "min": 2,
        "max": 30
    },
    "correct_spelling": False,
    "stemming": False,
    "remove_stopwords": False,
    "expand_contractions": True
}

benchmarks = {
    "accuracy":{},
    "cross_entropy":{},
    "predict_proba_predicted_character":{}
}

# src.classify_std.classify(technique="fasttext", C=10.0, max_iter=500, verbose=3)
test_options = ["test", "train"]
random_options = ["", "_random"]
techniques = ["tfidf", "fasttext"]

details = src.get_df("details_min_wordcount")
pd.set_option('display.float_format', '{:.2e}'.format)
d1 = {}
for test in test_options:
    details_cur = details["wordcount"][test]
    d2 = {}
    data = src.get_df("4_benchmark_change_testing_data_" + test + "")
    for tech in techniques:
        accuracy = data[tech]["accuracy"]
        std = accuracy*(1-accuracy)/details_cur
        d2.update({tech:std})
    df = pd.concat(d2, axis=1)
    d1.update({test:df})

main = pd.concat(d1, axis = 1)
print(main.to_latex())
# src.write_df(main, "dec", float_format='{:.2e}')