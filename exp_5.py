from math import sqrt
from statistics import mode
l=[[33.6,50,1],[26.6,30,0],[23.4,40,0],[43.1,67,0],[35.3,23,1],[35.9,67,1],[36.7,45,1],[25.7,46,0],[23.3,29,0],[31,56,1]]
n=[43.6,40]
k=3
m=[]
x=[]
for i in l:
    a=0
    for j in range(len(n)-1):
        a+= (i[j]-n[j])*(i[j]-n[j])
    m.append(sqrt(a))
a=sorted(m)
for i in range(k):
    x.append(m.index(a[i]))
y=[]
for i in x:
    print(l[i])
    y.append(l[i][-1])
print()
print("result -->",mode(y))
