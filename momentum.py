import csv

date=[]
price = []
mom_avg =[]
n=0
u_list = []
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




path = "/home/rohit/Downloads/stock-market/GOOGL.csv"
file = open(path)
csvreader = csv.reader(file)
header = next(csvreader)
for row in csvreader:
    date1 = int(row[0].split('-')[0])#you only get the years
    open_price = float(row[1])
    price.append(open_price)
    date.append(date1)

unique_list()

for i in u_list:
    mom_avg.append(calculate_avg_price_per_year(i))
avg = 0
l = 0
print(mom_avg)
for i in range(len(mom_avg)):
    avg = avg + mom_avg[i]
    l += 1

mom_factor = float(avg/l) - mom_avg[0]
print("the momentum factor in ",len(mom_avg)," year is ",mom_factor )


