#!/usr/bin/env python
# coding: utf-8

# # EN einhliðamálheild
# Í þessu reikniriti eru enskar einhliðamálheildir lesnar og staðlaðar svo ein setning sé í hverri línu. Málheildirnar eru eftirfrandi. Sameinum í eitt skjal.
# 
# | Nafn | #línur| stærð | ath |
# | --- | --- | --- | --- |
# | newscrawl 2018 en | 18,1M | 2 GB | - |
# | UN corpus EN | 27,4M | 3,4GB | - |
# | Europarl v9 | 2,3M | 321 MB | - |
# | Librispeech 27 | 48,2M | 4,3GB | - | - |
# | Wikipedia Google en | 25,8M | 3,0GB | - |
# | --- | --- | --- | --- |
# | Heild | 121,9M | 13,1GB | - |

# In[1]:
import sys
# The location of en-mono to read
source_dir = sys.argv[1]

# The location of where to write the results
target_dir = sys.argv[2]

from glob import glob
from pprint import pprint


# In[2]:


import pathlib

source_dir = pathlib.Path(source_dir)
target_dir = pathlib.Path(target_dir)
assert source_dir.exists()
if not target_dir.exists():
    target_dir.mkdir()


# In[3]:


def read_file(path):
    corpus = []
    with path.open() as f_in:
        return [line for line in f_in]


# ## Europarl EN
# Þessi gögn líta vel út og ekkert þarf að gera.

# In[4]:


euro_parl = source_dir.joinpath('europarl-v9.en')


# In[5]:


euro_parl = read_file(source_dir.joinpath('europarl-v9.en'))


# ## News crawl 2018
# Lítur vel út - ekkert sem þarf að gera.

# In[6]:


news = source_dir.joinpath('news.2018.en.shuffled.deduped')


# In[7]:


news = read_file(source_dir.joinpath('news.2018.en.shuffled.deduped'))


# ## Wikipedia
# Þessi gögn eru í `.csv`, við þurfum að laga það.

# In[8]:


import pandas as pd
wiki = source_dir.joinpath('documents_utf8_filtered_20pageviews.csv')
wiki_df = pd.read_csv(str(wiki), sep=",", header=None)
wiki_df.head()


# In[9]:


wiki = wiki_df[1].values.tolist()
wiki[:2]


# Núna er skjalið með eina grein í hverri línu.

# In[10]:


import nltk
import re

wiki_sent = []

for line in wiki:
    line = line.strip()
    for sent in nltk.sent_tokenize(line):
        wiki_sent.append(sent)


# In[11]:


wiki_sent[:5]


# ## UN Corpus en 
# Þessi málheild byggist á mörgum .tei skjölum. Við þurfum að lesa öll skjölin og sameina í eitt stórt skjal. Þar sem flest skjöl byrja á fundarlýsingu og öðrum setningum sem eru ekki sérlega góðar setningar. Við byrjum því að lesa setningar eftir setningu númer 40.

# In[12]:


un_dir = source_dir.joinpath('UNv1.0-TEI')
xml_files = glob(f'{un_dir}/**/*.xml', recursive=True)
xml_files = [pathlib.Path(xml_file) for xml_file in xml_files]
len(xml_files)


# In[13]:



# In[14]:


from xml.etree import ElementTree as ET

def read_un_tei_file(path: pathlib.Path, min_p_id=40):
    sentences = []
    try: 
        root = ET.parse(str(path)).getroot()
        # We gather all the paragraphs from the body, avoiding the divs
        for paragraph_node in root.findall('.//body//p'):
            # we only take sentences which have an "id" high enough.
            if int(paragraph_node.attrib['id']) < min_p_id:
                continue
            for sentence_node in paragraph_node.findall('.//s'):
                sentence = sentence_node.text
                sentences.append(sentence)
    except:
        # We just skip the file
        pass
    return sentences


# In[15]:


print(read_un_tei_file(xml_files[0]))


# In[16]:


un = []
for xml_file in xml_files:
    sentences = read_un_tei_file(xml_file, min_p_id=50)
    for sentence in sentences:
        if sentence:
            un.append(sentence + '\n')


# ## Libri Speech 27
# 
# Þessi málheild er samansafn af bókum og inniheldur mikið af tómum línum. Einnig er hver bók í sér skrá.

# In[17]:


libri_dir = source_dir.joinpath('librispeech').joinpath('corpus')
txt_files = glob(f'{libri_dir}/**/*.txt', recursive=True)
txt_files = [pathlib.Path(txt_file) for txt_file in txt_files]
len(txt_files)


# In[18]:




# In[19]:




# In[20]:


import nltk
import re

def read_libri_file(p_in):
    sentences = []
    buffer = []
    with p_in.open() as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                if buffer:
                    for sent in nltk.sent_tokenize(" ".join(buffer)):
                        sentences.append(sent)
                    buffer = []
            else:
                buffer.append(line)
    if buffer:
        for sent in nltk.sent_tokenize(" ".join(buffer)):
            sentences.append(sent)
        buffer = []
    return sentences


# In[21]:


read_libri_file(txt_files[0])[:16]


# Við náum ekki alveg textanum réttum með NLTK. En við köllum það gott.

# In[22]:


libri = []
for txt_file in txt_files:
    for sentence in read_libri_file(txt_file):
        libri.append(sentence + '\n')


# ## Sameinum í eitt skjal

# In[23]:


corpora = [
    euro_parl,
    news,
    wiki_sent,
    un,
    libri
]

with target_dir.joinpath('data.en').open('w+') as f_out:
    for corpus in corpora:
        for line in corpus:
            f_out.write(line)


# In[ ]:




