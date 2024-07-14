import sys
import os
import sqlite3

sys.path.append(os.getcwd())

from jpap.connect import get_company_postings
from jpap.ipl import IPL
from jpap.preprocessing import subsample_df

# get postings for a set of companies that were not in the training data (max. 5 postings per company)
jpod_conn = sqlite3.connect("/scicore/home/weder/GROUP/Innovation/05_job_adds_data/jpod_test.db")
companies = [
            "sanofi", "incyte", "vertex pharmaceuticals", "abbott laboratories", "baxter", "viatris", # pharma
            "georg fischer", "fresenius", "porsche", "adidas", "sonova", "richemont", "logitech", "ruag",# medtech/manufacturing
            "merrill lynch", "pictet",# banks & insurances
            "astreya", "western digital", "snapchat",# IT
            "utmb health", "hôpital du jura", # health
            "fachhochschule nordwestschweiz fhnw", # research
            "tibits", "grand hotel des bains kempinski st. moritz", "aloft hotels",# hotels & restaurants
            "sbb cff ffs", "20 minuten", "massachusetts department of transportation" # other
            ]
df = get_company_postings(con = jpod_conn, companies = companies, institution_name=True)
df = subsample_df(df=df, group_col="company_name", max_n_per_group=3).reset_index(drop=True)
company_names = df["company_name"].to_list()
postings_texts = df["job_description"].to_list()
assert len(company_names) == len(postings_texts)
print(f'"Predicting the industry association for {len(postings_texts)} postings of {len(set(company_names))} different companies"')

# load pipeline and predict all postings
INDUSTRY_LEVEL = "nace"
print(f'"Classification is performed at the following level: {INDUSTRY_LEVEL}"')
industry_pipeline = IPL(classifier = INDUSTRY_LEVEL)
df["industry"] = industry_pipeline(postings = postings_texts, company_names = company_names)

# majority vote for every companies
company_industry_labels = df.groupby(["company_name"]).apply(lambda x: x["industry"].value_counts().index[0]).to_dict()
for company, industry in company_industry_labels.items():
    print(f'"{company} is predicted to be part of the following industry: {industry}"')