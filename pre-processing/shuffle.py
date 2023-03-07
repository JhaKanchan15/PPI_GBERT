import random
lines = open('upd_Hprd/Hprd_Node.txt').readlines()
random.shuffle(lines)
open('upd_Hprd/Shuff_Hprd_Node.txt', 'w').writelines(lines)