import sys
import numpy as np

n = 0
mid = []
mid2 = []
mid3 = []
num1 = 0
num2 = 0
num11 = 0
num111 = 0
for line in open(sys.argv[1]):
    n += 1
    line = line.strip().split()
    if n%2 == 0:
        list1 = [float(i) for i in line]
        tt = list1[-2]
        mid.append(tt)
        mid2.append(list1[0])
        mid3.append(list1[2])
    if len(mid) ==  3:
        #print(mid.index(max(mid))+1, mid)
        if mid.index(max(mid)) == 0 and mid[0] != mid[1] and mid[0] != mid[-1]:
            num1 +=1
        if mid2.index(max(mid2)) == 0 and mid2[0] != mid2[1] and mid2[0] != mid2[-1]:
            num11 += 1
        if mid3.index(max(mid3)) == 0 and mid3[0] != mid3[1] and mid3[0] != mid3[-1]:
            num111 += 1
        num2 += 1
        mid = []
        mid2 = []
print('--------'+sys.argv[1]+'--------')
print('based on target_word pro')
print(float(num1)/num2)
print('based on cue_word pro')
print(float(num111)/num2)
print('based on sentence pro')
print(float(num11)/num2)

