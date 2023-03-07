# Concatenation of Positive and Negative edges to link.txt file

with open("Human/PositiveEdges.txt", "r") as f:
    lines = f.readlines()

with open("Human/NegativeEdges.txt", "r") as f:
    lines1 = f.readlines()

with open("Human/Human_Link.txt", "w") as myfile:
    for line in lines:
      myfile.write(line)
      
    #myfile.write('\n')  
    
    for line in lines1:
      myfile.write(line)