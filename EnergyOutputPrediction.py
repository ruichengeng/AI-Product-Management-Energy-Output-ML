import pandas as pd

EnergyDataFile = pd.read_csv(r"CCPP_data.csv")

KFoldDataDict = {}
IterationSize = 736
EnergyDataChunks = [EnergyDataFile[x:x+IterationSize] for x in range(0, len(EnergyDataFile), IterationSize)]

print ("hello this is the energy output prediction, count of the list is: ")
print (len(EnergyDataFile.index))

def PopulateDataDict():
    if (len(EnergyDataFile) == 9568):
        iterationSize = len(EnergyDataFile)
        print (EnergyDataChunks[0])

PopulateDataDict()

print (len(EnergyDataChunks))
print (EnergyDataChunks[0])
print ("End of Printing")
##print (len(KFoldDataDict))
##print (KFoldDataDict[1])