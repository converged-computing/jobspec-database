from evaluate import load
import pandas as pd
import numpy as np 
bertscore = load("bertscore")

gsm8k = pd.read_csv('data/gsm8k_questions.csv')
gsm8k = gsm8k.drop_duplicates(subset ='instruction')
gsm8k = pd.read_csv('data/gsm8k_original.csv')
mathwell_all = pd.read_csv('data/mathwell_annotations_final.csv')
mathwell_all_good = mathwell_all[mathwell_all['good']==1]
df = pd.read_csv('data/all_models.csv')
llama = df[df['model']=='llama']
llama_good = llama[llama['good']==1]
llema = df[df['model']=='llema']
llema_good = llema[llema['good']==1]
mathwell = df[df['model']=='mathwell']
mathwell_good = mathwell[mathwell['good']==1]
mammoth = df[df['model']=='mammoth']
mammoth_good = mammoth[mammoth['good']==1]
gpt35 = df[df['model']=='gpt35']
gpt35_good = gpt35[gpt35['good']==1]
gpt4 = df[df['model']=='gpt4']
gpt4_good = gpt4[gpt4['good']==1]
numglue = pd.read_csv('data/numglue_questions.csv')
numglue = numglue.drop_duplicates(subset = 'instruction')
asdiv = pd.read_csv('data/ASDiv_clean.csv')
svamp = pd.read_json('data/SVAMP.json')
svamp['question'] = svamp['Body'] + " " + svamp['Question']
gsm_hard = pd.read_json('data/gsmhard.json')
sgsm = pd.read_csv('sgsm.csv')
sgsm_unan = sgsm[sgsm['subset']=='sgsm_unannotated']
sgsm_train = sgsm[sgsm['subset']=='sgsm_train']
#sgsm = pd.concat([sgsm_unan, mathwell_all_good])

# def score(df1, df2, df1var, df2var, same_df = False, limit = 18000):
#     precision = []
#     recall = []
#     f1 = []
        
#     if same_df == False:
#         df1 = df1.sample(frac = 1)
#         df2 = df2.sample(frac = 1)
#         if limit > len(df1) or limit > len(df2):
#             lim1 = min(limit, len(df1))
#             lim2 = min(limit, len(df2))
            
#         else: 
#             lim1 = limit
#             lim2 = limit
            
#         for i in range(0, lim1):
#             for j in range(0, lim2):
#                 ref = df1.iloc[i][df1var]
#                 ref = str(ref)
#                 pred = df2.iloc[j][df2var]
#                 pred = str(pred)
#                 results = bertscore.compute(predictions=[pred], references=[ref], lang="en")
#                 precision.append(results['precision'])
#                 recall.append(results['recall'])
#                 f1.append(results['f1'])
                
#     if same_df == True:
#         if limit > len(df1) or limit > len(df2):
#             lim1 = min(limit, len(df1))
#             lim2 = min(limit, len(df2))
            
#         else: 
#             lim1 = limit
#             lim2 = limit
            
#         for i in range(0, lim1):
#             for j in range(len(df2)-lim2, len(df2)):
#                 ref = df1.iloc[i][df1var]
#                 ref = str(ref)
#                 pred = df2.iloc[j][df2var]
#                 pred = str(pred)
#                 results = bertscore.compute(predictions=[pred], references=[ref], lang="en")
#                 precision.append(results['precision'])
#                 recall.append(results['recall'])
#                 f1.append(results['f1'])
#     return (precision, recall, f1)

def score(df1, df2, df1var, df2var, same_df = False, limit = 2000):
    precision = []
    recall = []
    f1 = []
        
    if same_df == False:
        df1 = df1.sample(frac = 1)
        df2 = df2.sample(frac = 1)
        if limit > len(df1) or limit > len(df2):
            lim1 = min(limit, len(df1))
            lim2 = min(limit, len(df2))
            
        else: 
            lim1 = limit
            lim2 = limit
        
        refs = []
        preds = []
        for i in range(0, lim1):
            for j in range(0, lim2):
                ref = df1.iloc[i][df1var]
                ref = str(ref)
                pred = df2.iloc[j][df2var]
                pred = str(pred)
                preds.append(pred)
                refs.append(ref)
                if len(preds)==128:
                    results = bertscore.compute(predictions=preds, references=refs, lang="en", batch_size = 128)
                    precision.append(np.mean(results['precision']))
                    recall.append(np.mean(results['recall']))
                    f1.append(np.mean(results['f1']))
                    refs = []
                    preds = []
                
    if same_df == True:
        if limit > len(df1) or limit > len(df2):
            lim1 = min(limit, len(df1))
            lim2 = min(limit, len(df2))
            
        else: 
            lim1 = limit
            lim2 = limit
            
        refs = []
        preds = []
        for i in range(0, lim1):
            for j in range(len(df2)-lim2, len(df2)):
                ref = df1.iloc[i][df1var]
                ref = str(ref)
                pred = df2.iloc[j][df2var]
                pred = str(pred)
                preds.append(pred)
                refs.append(ref)
                if len(preds)==128:
                    results = bertscore.compute(predictions=preds, references=refs, lang="en", batch_size = 128)
                    precision.append(np.mean(results['precision']))
                    recall.append(np.mean(results['recall']))
                    f1.append(np.mean(results['f1']))
                    refs = []
                    preds = []
                    
    return (precision, recall, f1)
# scores = score(sgsm_train, sgsm_train, 'question', 'question', same_df = True)
# result = f"Average SGSM Train overall BERTScore: Precision: {np.mean(scores[0])}, Recall: {np.mean(scores[1])}, F1: {np.mean(scores[2])}"
# output_file = "bertscores.txt"  # Specify the path and filename for the output file
# with open(output_file, "a") as f:  # Open the file in append mode ("a")
#     f.write(result + "\n")  # Append the newly generated text to the file  # Append the newly generated text to the file
    
# scores = score(sgsm_unan, sgsm_unan, 'question', 'question', same_df = True)
# result = f"Average SGSM Unannotated overall BERTScore: Precision: {np.mean(scores[0])}, Recall: {np.mean(scores[1])}, F1: {np.mean(scores[2])}"
# output_file = "bertscores.txt"  # Specify the path and filename for the output file
# with open(output_file, "a") as f:  # Open the file in append mode ("a")
#     f.write("\n\n ### New Results ### \n\n" + result + "\n")  # Append the newly generated text to the file

scores = score(sgsm_unan, gsm8k, 'question', 'question')
result = f"Average SGSM Unannotated/GSM8K overall BERTScore: Precision: {np.mean(scores[0])}, Recall: {np.mean(scores[1])}, F1: {np.mean(scores[2])}"
output_file = "bertscores.txt"  # Specify the path and filename for the output file
with open(output_file, "a") as f:  # Open the file in append mode ("a")
    f.write(result + "\n")  # Append the newly generated text to the file
    
# scores = score(sgsm_unan, sgsm_train, 'question', 'question')
# result = f"Average SGSM Unannotated/SGSM Train overall BERTScore: Precision: {np.mean(scores[0])}, Recall: {np.mean(scores[1])}, F1: {np.mean(scores[2])}"
# output_file = "bertscores.txt"  # Specify the path and filename for the output file
# with open(output_file, "a") as f:  # Open the file in append mode ("a")
#     f.write(result + "\n")  # Append the newly generated text to the file  # Append the newly generated text to the file

# scores = score(sgsm, sgsm, 'question', 'question', same_df = True)
# result = f"Average SGSM overall BERTScore: Precision: {np.mean(scores[0])}, Recall: {np.mean(scores[1])}, F1: {np.mean(scores[2])}"
# output_file = "bertscores.txt"  # Specify the path and filename for the output file
# with open(output_file, "a") as f:  # Open the file in append mode ("a")
#     f.write(result + "\n")  # Append the newly generated text to the file
    
scores = score(gsm8k, gsm8k, 'question', 'question', same_df = True)
result = f"Average GSM8K overall BERTScore: Precision: {np.mean(scores[0])}, Recall: {np.mean(scores[1])}, F1: {np.mean(scores[2])}"
output_file = "bertscores.txt"  # Specify the path and filename for the output file
with open(output_file, "a") as f:  # Open the file in append mode ("a")
    f.write(result + "\n")  # Append the newly generated text to the file
    
# scores = score(mathwell_all_good, mathwell_all_good, 'question', 'question', same_df = True)
# result = f"Average MATHWELL Train overall BERTScore: Precision: {np.mean(scores[0])}, Recall: {np.mean(scores[1])}, F1: {np.mean(scores[2])}"
# output_file = "bertscores.txt"  # Specify the path and filename for the output file
# with open(output_file, "a") as f:  # Open the file in append mode ("a")
#     f.write(result + "\n")  # Append the newly generated text to the file

# scores = score(mathwell_all, mathwell_all, 'question', 'question', same_df = True)
# result = f"Average MATHWELL Annotated overall BERTScore: Precision: {np.mean(scores[0])}, Recall: {np.mean(scores[1])}, F1: {np.mean(scores[2])}"
# output_file = "bertscores.txt"  # Specify the path and filename for the output file
# with open(output_file, "a") as f:  # Open the file in append mode ("a")
#     f.write(result + "\n")  # Append the newly generated text to the file

# scores = score(mathwell_all, mathwell_all_good, 'question', 'question', same_df = True)
# result = f"Average MATHWELL Annotated/MATHWELL Train BERTScore: Precision: {np.mean(scores[0])}, Recall: {np.mean(scores[1])}, F1: {np.mean(scores[2])}"
# output_file = "bertscores.txt"  # Specify the path and filename for the output file
# with open(output_file, "a") as f:  # Open the file in append mode ("a")
#     f.write(result + "\n")  # Append the newly generated text to the file
    
scores = score(sgsm_train, gsm8k, 'question', 'question')
result = f"Average SGSM Train/GSM8K overall BERTScore: Precision: {np.mean(scores[0])}, Recall: {np.mean(scores[1])}, F1: {np.mean(scores[2])}"
output_file = "bertscores.txt"  # Specify the path and filename for the output file
with open(output_file, "a") as f:  # Open the file in append mode ("a")
    f.write(result + "\n")  # Append the newly generated text to the file
    
# scores = score(mathwell_all, gsm8k, 'question', 'question')
# result = f"Average MATHWELL Annotated/GSM8K overall BERTScore: Precision: {np.mean(scores[0])}, Recall: {np.mean(scores[1])}, F1: {np.mean(scores[2])}"
# output_file = "bertscores.txt"  # Specify the path and filename for the output file
# with open(output_file, "a") as f:  # Open the file in append mode ("a")
#     f.write(result + "\n")  # Append the newly generated text to the file
    
# scores = score(mathwell, mathwell, 'question', 'question', same_df = True, limit = 250)
# result = f"Average MATHWELL overall BERTScore: Precision: {np.mean(scores[0])}, Recall: {np.mean(scores[1])}, F1: {np.mean(scores[2])}"
# output_file = "bertscores.txt"  # Specify the path and filename for the output file
# with open(output_file, "a") as f:  # Open the file in append mode ("a")
#     f.write(result + "\n")  # Append the newly generated text to the file
    
# scores = score(mathwell_good, mathwell_good, 'question', 'question', same_df = True, limit = 250)
# result = f"Average MATHWELL MaC overall BERTScore: Precision: {np.mean(scores[0])}, Recall: {np.mean(scores[1])}, F1: {np.mean(scores[2])}"
# output_file = "bertscores.txt"  # Specify the path and filename for the output file
# with open(output_file, "a") as f:  # Open the file in append mode ("a")
#     f.write(result + "\n")  # Append the newly generated text to the file
    
# # scores = score(mathwell_good, mathwell_all_good, 'question', 'question')
# # result = f"Average MATHWELL MaC/MATHWELL Train overall BERTScore: Precision: {np.mean(scores[0])}, Recall: {np.mean(scores[1])}, F1: {np.mean(scores[2])}"
# # output_file = "bertscores.txt"  # Specify the path and filename for the output file
# # with open(output_file, "a") as f:  # Open the file in append mode ("a")
# #     f.write(result + "\n")  # Append the newly generated text to the file

# scores = score(mathwell_good, mathwell, 'question', 'question')
# result = f"Average MATHWELL MaC/MATHWELL all generations BERTScore: Precision: {np.mean(scores[0])}, Recall: {np.mean(scores[1])}, F1: {np.mean(scores[2])}"
# output_file = "bertscores.txt"  # Specify the path and filename for the output file
# with open(output_file, "a") as f:  # Open the file in append mode ("a")
#     f.write(result + "\n")  # Append the newly generated text to the file
    
scores = score(gsm8k, mathwell, 'question', 'question')
result = f"Average MATHWELL/GSM8K BERTScore: Precision: {np.mean(scores[0])}, Recall: {np.mean(scores[1])}, F1: {np.mean(scores[2])}"
output_file = "bertscores.txt"  # Specify the path and filename for the output file
with open(output_file, "a") as f:  # Open the file in append mode ("a")
    f.write(result + "\n")  # Append the newly generated text to the file

# scores = score(gpt35, gpt35, 'question', 'question', same_df = True, limit = 250)
# result = f"Average gpt35 overall BERTScore: Precision: {np.mean(scores[0])}, Recall: {np.mean(scores[1])}, F1: {np.mean(scores[2])}"
# output_file = "bertscores.txt"  # Specify the path and filename for the output file
# with open(output_file, "a") as f:  # Open the file in append mode ("a")
#     f.write(result + "\n")  # Append the newly generated text to the file

scores = score(gsm8k, gpt35, 'question', 'question')
result = f"Average gpt35 MaC/GSM8K BERTScore: Precision: {np.mean(scores[0])}, Recall: {np.mean(scores[1])}, F1: {np.mean(scores[2])}"
output_file = "bertscores.txt"  # Specify the path and filename for the output file
with open(output_file, "a") as f:  # Open the file in append mode ("a")
    f.write(result + "\n")  # Append the newly generated text to the file
    
# scores = score(gpt4, gpt4, 'question', 'question', same_df = True, limit = 250)
# result = f"Average gpt4 overall BERTScore: Precision: {np.mean(scores[0])}, Recall: {np.mean(scores[1])}, F1: {np.mean(scores[2])}"
# output_file = "bertscores.txt"  # Specify the path and filename for the output file
# with open(output_file, "a") as f:  # Open the file in append mode ("a")
#     f.write(result + "\n")  # Append the newly generated text to the file

scores = score(gsm8k, gpt4, 'question', 'question')
result = f"Average gpt4 MaC/GSM8K BERTScore: Precision: {np.mean(scores[0])}, Recall: {np.mean(scores[1])}, F1: {np.mean(scores[2])}"
output_file = "bertscores.txt"  # Specify the path and filename for the output file
with open(output_file, "a") as f:  # Open the file in append mode ("a")
    f.write(result + "\n")  # Append the newly generated text to the file

# scores = score(gpt35_good, gpt35_good, 'question', 'question', same_df = True, limit = 250)
# result = f"Average gpt35 MaC overall BERTScore: Precision: {np.mean(scores[0])}, Recall: {np.mean(scores[1])}, F1: {np.mean(scores[2])}"
# output_file = "bertscores.txt"  # Specify the path and filename for the output file
# with open(output_file, "a") as f:  # Open the file in append mode ("a")
#     f.write(result + "\n")  # Append the newly generated text to the file
    
# scores = score(gpt35,gpt35_good, 'question', 'question')
# result = f"Average gpt35 all generations/gpt35 MaC overall BERTScore: Precision: {np.mean(scores[0])}, Recall: {np.mean(scores[1])}, F1: {np.mean(scores[2])}"
# output_file = "bertscores.txt"  # Specify the path and filename for the output file
# with open(output_file, "a") as f:  # Open the file in append mode ("a")
#     f.write(result + "\n")  # Append the newly generated text to the file

# scores = score(gpt4_good, gpt4_good, 'question', 'question', same_df = True, limit = 250)
# result = f"Average gpt4 MaC overall BERTScore: Precision: {np.mean(scores[0])}, Recall: {np.mean(scores[1])}, F1: {np.mean(scores[2])}"
# output_file = "bertscores.txt"  # Specify the path and filename for the output file
# with open(output_file, "a") as f:  # Open the file in append mode ("a")
#     f.write(result + "\n")  # Append the newly generated text to the file
    
# scores = score(gpt4,gpt4_good, 'question', 'question')
# result = f"Average gpt4 all generations/gpt35 MaC overall BERTScore: Precision: {np.mean(scores[0])}, Recall: {np.mean(scores[1])}, F1: {np.mean(scores[2])}"
# output_file = "bertscores.txt"  # Specify the path and filename for the output file
# with open(output_file, "a") as f:  # Open the file in append mode ("a")
#     f.write(result + "\n")  # Append the newly generated text to the file
# scores = score(llama, llama, 'question', 'question', same_df = True, limit = 250)
# result = f"Average llama overall BERTScore: Precision: {np.mean(scores[0])}, Recall: {np.mean(scores[1])}, F1: {np.mean(scores[2])}"
# output_file = "bertscores.txt"  # Specify the path and filename for the output file
# with open(output_file, "a") as f:  # Open the file in append mode ("a")
#     f.write(result + "\n")  # Append the newly generated text to the file

# scores = score(llama_good, llama_good, 'question', 'question', same_df = True, limit = 250)
# result = f"Average llama MaC overall BERTScore: Precision: {np.mean(scores[0])}, Recall: {np.mean(scores[1])}, F1: {np.mean(scores[2])}"
# output_file = "bertscores.txt"  # Specify the path and filename for the output file
# with open(output_file, "a") as f:  # Open the file in append mode ("a")
#     f.write(result + "\n")  # Append the newly generated text to the file
    
# scores = score(llama,llama_good, 'question', 'question')
# result = f"Average llama all generations/llama MaC overall BERTScore: Precision: {np.mean(scores[0])}, Recall: {np.mean(scores[1])}, F1: {np.mean(scores[2])}"
# output_file = "bertscores.txt"  # Specify the path and filename for the output file
# with open(output_file, "a") as f:  # Open the file in append mode ("a")
#     f.write(result + "\n")  # Append the newly generated text to the file

scores = score(gsm8k, llama, 'question', 'question')
result = f"Average llama MaC/GSM8K BERTScore: Precision: {np.mean(scores[0])}, Recall: {np.mean(scores[1])}, F1: {np.mean(scores[2])}"
output_file = "bertscores.txt"  # Specify the path and filename for the output file
with open(output_file, "a") as f:  # Open the file in append mode ("a")
    f.write(result + "\n")  # Append the newly generated text to the file
    
# scores = score(llema, llema, 'question', 'question', same_df = True, limit = 250)
# result = f"Average llema overall BERTScore: Precision: {np.mean(scores[0])}, Recall: {np.mean(scores[1])}, F1: {np.mean(scores[2])}"
# output_file = "bertscores.txt"  # Specify the path and filename for the output file
# with open(output_file, "a") as f:  # Open the file in append mode ("a")
#     f.write(result + "\n")  # Append the newly generated text to the file
    
# scores = score(llema_good, llema_good, 'question', 'question', same_df = True, limit = 250)
# result = f"Average llema MaC overall BERTScore: Precision: {np.mean(scores[0])}, Recall: {np.mean(scores[1])}, F1: {np.mean(scores[2])}"
# output_file = "bertscores.txt"  # Specify the path and filename for the output file
# with open(output_file, "a") as f:  # Open the file in append mode ("a")
#     f.write(result + "\n")  # Append the newly generated text to the file
    
# scores = score(llema,llema_good, 'question', 'question')
# result = f"Average llema all generations/llema MaC overall BERTScore: Precision: {np.mean(scores[0])}, Recall: {np.mean(scores[1])}, F1: {np.mean(scores[2])}"
# output_file = "bertscores.txt"  # Specify the path and filename for the output file
# with open(output_file, "a") as f:  # Open the file in append mode ("a")
#     f.write(result + "\n")  # Append the newly generated text to the file

scores = score(gsm8k, llema, 'question', 'question')
result = f"Average llema MaC/GSM8K BERTScore: Precision: {np.mean(scores[0])}, Recall: {np.mean(scores[1])}, F1: {np.mean(scores[2])}"
output_file = "bertscores.txt"  # Specify the path and filename for the output file
with open(output_file, "a") as f:  # Open the file in append mode ("a")
    f.write(result + "\n")  # Append the newly generated text to the file
    
# scores = score(mammoth, mammoth, 'question', 'question', same_df = True, limit = 250)
# result = f"Average mammoth overall BERTScore: Precision: {np.mean(scores[0])}, Recall: {np.mean(scores[1])}, F1: {np.mean(scores[2])}"
# output_file = "bertscores.txt"  # Specify the path and filename for the output file
# with open(output_file, "a") as f:  # Open the file in append mode ("a")
#     f.write(result + "\n")  # Append the newly generated text to the file
    
# scores = score(mammoth_good, mammoth_good, 'question', 'question', same_df = True, limit = 250)
# result = f"Average mammoth MaC overall BERTScore: Precision: {np.mean(scores[0])}, Recall: {np.mean(scores[1])}, F1: {np.mean(scores[2])}"
# output_file = "bertscores.txt"  # Specify the path and filename for the output file
# with open(output_file, "a") as f:  # Open the file in append mode ("a")
#     f.write(result + "\n")  # Append the newly generated text to the file
    
# scores = score(mammoth, mammoth_good, 'question', 'question')
# result = f"Average mammoth all generations/mammoth MaC overall BERTScore: Precision: {np.mean(scores[0])}, Recall: {np.mean(scores[1])}, F1: {np.mean(scores[2])}"
# output_file = "bertscores.txt"  # Specify the path and filename for the output file
# with open(output_file, "a") as f:  # Open the file in append mode ("a")
#     f.write(result + "\n")  # Append the newly generated text to the file

scores = score(gsm8k, mammoth, 'question', 'question')
result = f"Average mammoth MaC/GSM8K BERTScore: Precision: {np.mean(scores[0])}, Recall: {np.mean(scores[1])}, F1: {np.mean(scores[2])}"
output_file = "bertscores.txt"  # Specify the path and filename for the output file
with open(output_file, "a") as f:  # Open the file in append mode ("a")
    f.write(result + "\n")  # Append the newly generated text to the file
    
# scores = score(numglue, numglue, 'instruction', 'instruction', same_df = True)
# result = f"Average numglue overall BERTScore: Precision: {np.mean(scores[0])}, Recall: {np.mean(scores[1])}, F1: {np.mean(scores[2])}"
# output_file = "bertscores.txt"  # Specify the path and filename for the output file
# with open(output_file, "a") as f:  # Open the file in append mode ("a")
#     f.write(result + "\n")  # Append the newly generated text to the file
    #numglue is instruction rest are questions
    
# # scores = score(asdiv, asdiv, 'question', 'question', same_df = True)
# # result = f"Average asdiv overall BERTScore: Precision: {np.mean(scores[0])}, Recall: {np.mean(scores[1])}, F1: {np.mean(scores[2])}"
# # output_file = "bertscores.txt"  # Specify the path and filename for the output file
# # with open(output_file, "a") as f:  # Open the file in append mode ("a")
# #     f.write(result + "\n")  # Append the newly generated text to the file
    
# # scores = score(svamp, svamp, 'question', 'question', same_df = True)
# # result = f"Average svamp overall BERTScore: Precision: {np.mean(scores[0])}, Recall: {np.mean(scores[1])}, F1: {np.mean(scores[2])}"
# # output_file = "bertscores.txt"  # Specify the path and filename for the output file
# # with open(output_file, "a") as f:  # Open the file in append mode ("a")
# #     f.write(result + "\n")  # Append the newly generated text to the file
    
# # scores = score(gsm_hard, gsm_hard, 'question', 'question', same_df = True)
# # result = f"Average gsmhard overall BERTScore: Precision: {np.mean(scores[0])}, Recall: {np.mean(scores[1])}, F1: {np.mean(scores[2])}"
# # output_file = "bertscores.txt"  # Specify the path and filename for the output file
# # with open(output_file, "a") as f:  # Open the file in append mode ("a")
# #     f.write(result + "\n")  # Append the newly generated text to the file