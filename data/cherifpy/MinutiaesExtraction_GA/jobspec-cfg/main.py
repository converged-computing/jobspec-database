from geneticalgorithme import *
from selection import SelectBestSolution
from parametres import *
from data import *
import datetime
from keras.optimizers import SGD, Adam
import os
import sys


if __name__ == "__main__":

    
    

    date = datetime.datetime.now()
    date = date.strftime("%m_%d_%H_%M_%S")


    PATHS = {
        "TextFile":f"Tests/file_{date}.txt",
        "CSVFile":f"Tests/file_{date}.csv",
        "MemorieFile":"Tests/memorie.json",
        "ResultsFile":"Tests/results.json",
        "BestFile":"Tests/BestIndiv.json"
    }

    f = open(PATHS["TextFile"],"w")
    f.write(f"NB_OF_GENERATION = {NB_OF_GENERATION}\nPOPULATION_SIZE = {POPULATION_SIZE}\nINPUT_SHAPE "+
            f"= {INPUT_SHAPE}\nBATCH_SIZE = {BATCH_SIZE}\nNB_EPOCHS = {NB_EPOCHS}\nELITE_FRAC = {ELITE_FRAC}\nCHILDREN_FRAC = "+
           f"{CHILDREN_FRAC}\nTEST_SIZE = {TEST_SIZE}\nPROBA_MUTATION = {PROBA_MUTATION}\n"+
           f"PROBA_CROSSOVER = {PROBA_CROSSOVER}\nVERSIONEN_CODAGE = {VERSION_ENCODAGE}\n")

    f.close()

    if not os.path.isfile(PATHS["CSVFile"]):
        columns = ["train accuracy", "test accuracy", "time"]
        csv_file = open(PATHS["CSVFile"], 'w',newline='')
        writer = csv.DictWriter(csv_file,fieldnames=columns)
        writer.writeheader()
        csv_file.close()


    if MODEL_MUN == 1:
        TrainSetPath = sys.argv[1]
        TestSetPath = sys.argv[2]
        TrainSet,TestSet = LoadDataBaseM1(TrainSetPath,TestSetPath,BATCH_SIZE,INPUT_SHAPE[0])

        Database = [[TrainSet],[TestSet]]
    else:
        csv_file = sys.argv[1]
        image_path = sys.argv[2] 
        TrainSet_X,TestSet_X,TrainSet_Y, TestSet_Y = LoadDataBaseM2(csv_file,image_path,TestSplit=TEST_SIZE)
        Database = [[TrainSet_X,TrainSet_Y],[TestSet_X,TestSet_Y]]


    optimizer = SGD(LEARNING_RATE)
    best_solution = GeneticAlgorithme(VERSION_ENCODAGE,POPULATION_SIZE,NB_OF_GENERATION,PROBA_PARENTS,
                                      ELITE_FRAC,CHILDREN_FRAC,optimizer,INPUT_SHAPE,Database,NB_EPOCHS,
                                      BATCH_SIZE,PROBA_CROSSOVER,PROBA_MUTATION,PATHS,MODEL_MUN)
    
    
    data = {}
    for indiv in best_solution:
        newkey =  f"individual{len(data)+1}"

        data[newkey] = {
            "individual": indiv[0],
            "fitness":indiv[1],
        }

    with open(PATHS["BestFile"],"w") as file:
        json.dump(data,file)
    file.close()
