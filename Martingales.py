import csv
import math

price = []
date = []
u_list = []
mom_avg =[]
s_mom_avg = []
n=0
sum_y = 0
p = 0.00

def calculate_avg_price_per_year(x):
    avg = 0
    n = 0
    for i in range(len(date)):
        if(date[i] == int(x)):
            avg = float(avg + price[i])
            n = n+1
    l= float(avg/n)
    return l

def unique_list():
    for x in date:
        if x not in u_list:
            u_list.append(x)

path = "/home/rohit/Downloads/stock-market/AMD.csv"
file = open(path)
csvreader = csv.reader(file)
header = next(csvreader)
for row in csvreader:
    date1 = int(row[0].split('-')[0])
    open_price = float(row[1])
    price.append(open_price)
    date.append(date1)

unique_list()

for i in u_list:
    mom_avg.append(calculate_avg_price_per_year(i))



for i in range(len(mom_avg)):
    p = float(mom_avg[i])
    p = float(mom_avg[i] * mom_avg[i])
    s_mom_avg.append(p)


print(mom_avg)

sum_y = sum(s_mom_avg)
sum_y = sum_y/len(mom_avg)
sum_y = math.sqrt(sum_y)
print(sum_y)

