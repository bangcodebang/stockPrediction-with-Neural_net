import csv

c_price = []
volume = []
p_e = []
net_profit =[]
earning =[]



path = "/home/rohit/Downloads/stock-market/AMD.csv"
file = open(path)
csvreader = csv.reader(file)
header = next(csvreader)
for row in csvreader:
    open_price = float(row[1])
    volume_stock = float(row[6])
    closing_price = float(row[4])
    c_price.append(open_price)
    volume.append(volume_stock)
    net_profit.append((closing_price - open_price) * volume_stock)

for i in range(len(net_profit)):
    if(net_profit[i] == 0):
        earning.append(0)
    else:
        earning.append(float(net_profit[i]/volume[i]))


for i in range(len(earning)):
    if(earning[i]<0):
        l = float(c_price[i] / -1*earning[i])
        l= -1*l;
        p_e.append(float(c_price[i] / l))
    elif(earning[i] == 0):
        p_e.append(0)
    else:
        p_e.append(float(c_price[i] / earning[i]))


if(p_e[-1] <= 150  ):
    flag = 1
else:
    flag = 0

print(flag)


