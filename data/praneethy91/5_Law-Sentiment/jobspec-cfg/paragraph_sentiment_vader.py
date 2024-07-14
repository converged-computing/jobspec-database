# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 13:14:22 2017

@author: naman

To start the server:
java -mx5g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -timeout 1000000000000000000000000000000000000000

timeout can be changed
"""

from pycorenlp import StanfordCoreNLP
import utils as util
import sys
import codecs
import time
from nltk import tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

if sys.stdout.encoding != 'cp850':
  sys.stdout = codecs.getwriter('cp850')(sys.stdout.buffer, 'strict')
if sys.stderr.encoding != 'cp850':
  sys.stderr = codecs.getwriter('cp850')(sys.stderr.buffer, 'strict')

demo = False
demo_local = False
sid = SentimentIntensityAnalyzer()
outDir = "/home/nk2239/VADER_DATA_STORE"

def getParaSentimentList(para_list):
    avgParaSentimentList = []
    paraSentimentList = []
    #start = time.time()
    for para in para_list:

        sentence_list = tokenize.sent_tokenize(para)
        para_sentiment = 0
        sentence_count = 0
        #if(demo or demo_local):
            #print(para)
            #print(res)
        for sentence in sentence_list:
            #if(demo or demo_local):
                #print(sentence)
            sentence_sentiment = sid.polarity_scores(sentence)
            para_sentiment += sentence_sentiment['compound']
            sentence_count += 1
        if(sentence_count == 0):
            avgParaSentimentList.append(0)
        else:
            avgParaSentimentList.append(para_sentiment)
        sentiment = 1 if para_sentiment > 0 else (-1 if para_sentiment < 0 else 0)
        paraSentimentList.append(sentiment)
    #end = time.time()
    #print(end - start)
    return avgParaSentimentList, paraSentimentList


def main():
    if(not demo):
        root_Directory = 'data/clean_Mar_20'
        if(demo_local):
            global outDir
            outDir = "VADER_sentiment"
            root_Directory = '../../Data/clean_Mar_20'
        list_of_dirs = util.getDirectoryList(root_Directory)
        for directory in list_of_dirs:
            if not directory.endswith('zip'):
                year = int(directory)
                if year >= 1964:
                    print(directory)
                    util.createDirectory(outDir)
                    util.createDirectory(outDir + "/"+ directory)

                    files = util.getFilesListFromDir(directory)
                    for file_name in files:
                        para_list = util.getParaListFromFile(file_name, directory)
                        avgParaSentimentList, paraSentimentList = getParaSentimentList(para_list)
                        util.writeToPickle(paraSentimentList, outDir, directory, file_name, avg=False)
                        util.writeToPickle(avgParaSentimentList, outDir, directory, file_name, avg=True)
    else:
        para_list = [
            "DRUMMOND, C. J. The schooner American was at Oswego in the fall of 1872, and took in a cargo of coal for Chicago, leaving Oswego on the tenth of November. A general bill of lading was given, and a high price charged for the transportation of the coal from Oswego to Chicago, being $2.75 per ton. The schooner met with adverse winds and did not arrive at Port Huron until November 29th. The weather, according to the testimony of the witnesses, was very inclement that fall, and the captain concluded that the safest course was to strip the vessel and lay up at Port Huron. The schooner accordingly remained there with her cargo during the winter, and the coal was not delivered in Chicago or received by the consignees until May 8, 1873, when the spring freight was paid by the consignees on the coal, being much less than that charged in the bill of lading. After the coal had been thus delivered by the schooner to the consignees, a libel was filed claiming the amount of freight stated in the bill of lading, the consignees having refused to pay any more than the spring price of freight. The case went to proof before the district court, where the libel was dismissed; but a cross-libel having been filed claiming that the captain of the American was negligent in wintering at Port Hur on, and that the vessel should have come on in the fall of 1872, the district court gave a decree on the cross-libel for damages against the libelants in consequence of the supposed negligence of the captain. From t hese decrees the libelants have appealed to this court, and the question is whether the decrees of the district court are right.",
            "Several cities, New York City in particular for this paper, have a 311 24-hour hot line and online service, which allows anyone, residents and tourists, to report a non-emergency problem. Reported 311 problems are passed along to government services, who address and solve the problem. The records of 311 calls are publicly open and updated daily.",
            "Analysis of 311 calls can clearly be of great use for a wide variety of purposes, ranging from a rich understanding of the status of a city to the effectiveness of the government services in addressing such calls. Ideally, the analysis can also support a prediction of future 311 calls, which would enable the assignment of service resources by the city government.",
            "We have been extensively analyzing 311 calls in NYC. In this paper, we profile the data set and highlight a few interesting facts. We provide statistics along complaint types, geolocation, and temporal patterns and show the diversity of the big 311 data along those dimensions. We then discuss the prediction problem of number of calls, where we experiment with different sets of semantic features. We show that the prediction error for different complaint types can significantly vary if some features are not considered."]
        avgParaSentimentList, paraSentimentList = getParaSentimentList(para_list)
        print(avgParaSentimentList)
        print(paraSentimentList)


if __name__ == "__main__":
    main()