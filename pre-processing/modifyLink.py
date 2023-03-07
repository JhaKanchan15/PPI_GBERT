
with open("/Data/kanchan_1821cs15/sourav/Graph-Bert/Hprd_Link.txt", "r") as f:
    lines = f.readlines()
    print('\nTotal no. of lines: ',len(lines))
    
    
totCount = 0
lineCount = 0

with open("/Data/kanchan_1821cs15/sourav/Graph-Bert/upd_Hprd/new_Hprd_link.txt", "w") as f:
    for line in lines:
      print('Count = ',lineCount)
      
#      Added for test only
#      if count==5:
#        break
      
      arr = (line.strip()).split('\t')
      id1 = arr[1] + arr[0]
      
      if lineCount<len(lines)-1:
        nextArr = ((lines[lineCount+1]).strip()).split('\t')
      
        tempCnt = lineCount  
        span = 5
        while(arr[0] == nextArr[0] and tempCnt<lineCount+span):
          id2 = nextArr[1] + nextArr[0]
          newLine = id1 + '\t' + id2 + '\n'
          f.write(newLine)
          
          if tempCnt<len(lines)-1:
            nextArr = ((lines[tempCnt+1]).strip()).split('\t')
            
          tempCnt = tempCnt + 1
        
        totCount = totCount + tempCnt
        
      lineCount = lineCount + 1
      
print('Total no. of modified edges/links = ',totCount)