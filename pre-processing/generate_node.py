# Node : id + feature vector + class label
# In testing phase, need to structure all the parts and 
# have to consider all the edge cases possible


import numpy as np 

data = np.load('seqvec_files/hprd_seqvec_dict.npy', allow_pickle=True).tolist()

with open("Hprd_Link.txt", "r") as f:
    Lines = f.readlines()
    
with open("proteinList.txt", "r") as f:
    protLines = f.readlines()
        
count = 1
nodeCnt = 0
withotfeatpIDS = []
posCount = 0
negCount = 0

with open('upd_Hprd/Hprd_Node.txt', 'w') as f:
    for line in Lines:        
        
        #print(count)
        # Added for test only
#        if count==5:
#          break
              
        arr = (line.strip()).split('\t')
        #print(arr)
        prot1 = int(arr[0])
        prot2 = int(arr[1])       
        newID = arr[1] + arr[0]
        #print(newID)
        
        protIDArray1 = (protLines[prot1].strip()).split('\t')
        key1 = protIDArray1[1]       
        if data.get(key1) is None:
          withotfeatpIDS.append(prot1)
          continue
        
        feat1 = data.get(key1).tolist()
        
        protIDArray2 = (protLines[prot2].strip()).split('\t')
        key2 = protIDArray2[1]
        if data.get(key2) is None:
          withotfeatpIDS.append(prot2)
          continue
          
        feat2 = data.get(key2).tolist()
        
        newFeat = feat1 + feat2
        #print(feat1)
        #print(feat2)
        #print(newFeat)
        #print('Length of feature vector 1 is ', len(feat1))
        #print('Length of feature vector 2 is ', len(feat2))
        print('New length of feature vector is ', len(newFeat))
        
        
        print('Count = ',count,' New ID = ',newID)
        if count<=36557:
          label = 'Positive'
          posCount = posCount + 1
        else:
          label = 'Negative'
          negCount = negCount + 1
        
        
        if count<=5000 or count>36557:
          newLine = str(newID) + '\t' + str(newFeat).strip('[]').replace(', ','\t') + '\t' + label + '\n'
          nodeCnt = nodeCnt + 1
          f.write(newLine)
          
        count = count + 1

print('IDs without feature vector:',withotfeatpIDS)
print('Total count = ',count-1)
print('\nTotal node count = ',nodeCnt)
print('\n\n Positive Nodes = ',posCount, '\t\t Negative Nodes = ',negCount)
      
          