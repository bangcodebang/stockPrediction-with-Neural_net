import csv
import math

price = []
date = []
devation = []

def calculate_avg():
    avg = 0
    n = 0
    for i in range(len(price)):
            avg = float(avg + price[i])
            n = n+1
    l= float(avg/n)
    return l




path = "/home/rohit/Downloads/stock-market/IEX_API.csv"
file = open(path)
csvreader = csv.reader(file)
header = next(csvreader)
for row in csvreader:
    open_price = float(row[1])
    price.append(open_price)

avg1 = calculate_avg()
high_price = max(price)
l = price.index(high_price)
n=len(price)
a=0
for i in range(l,n):
    if(price[i] == avg1 or price[i] < avg1):
        a=1
if(a==1):
    print("mean reve")
else:
    print("not yet")