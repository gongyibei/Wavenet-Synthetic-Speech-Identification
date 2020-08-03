import sys

def creat_label(m,n):
    lable = ['1\n']*m + ['0\n']*n
    with open('label.txt','w') as f:
        f.writelines(lable)
