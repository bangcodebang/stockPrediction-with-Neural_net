import csv
import math
import numpy as np
import time


price = []
date = []
u_list = []
mom_avg =[]
s_mom_avg = []
devation = []
c_price = []
volume = []
p_e = []
net_profit =[]
earning =[]

def clear_list():
    price.clear()
    date.clear()
    u_list.clear()
    mom_avg.clear()
    s_mom_avg.clear()
    devation.clear()
    c_price.clear()
    volume.clear()
    p_e.clear()
    net_profit.clear()
    earning.clear()

def get_data(s):
    path = "/home/rohit/Downloads/stock-market/" + s
    file = open(path)
    csvreader = csv.reader(file)
    header = next(csvreader)
    for row in csvreader:
        date1 = int(row[0].split('-')[0])#you only get the years
        open_price = float(row[1])
        price.append(open_price)
        date.append(date1)
        volume_stock = float(row[6])
        closing_price = float(row[4])
        c_price.append(open_price)
        volume.append(volume_stock)
        net_profit.append((closing_price - open_price) * volume_stock)


def unique_list():
    for x in date:
        if x not in u_list:
            u_list.append(x)

def calculate_avg_price_per_year(x):
    avg = 0
    n = 0
    for i in range(len(date)):
        if(date[i] == int(x)):
            avg = float(avg + price[i])
            n = n+1
    l= float(avg/n)
    return l

def calculate_avg():
    avg = 0
    n = 0
    for i in range(len(price)):
            avg = float(avg + price[i])
            n = n+1
    l= float(avg/n)
    return l


def cal_momentum():
    unique_list()
    for i in u_list:
        mom_avg.append(calculate_avg_price_per_year(i))
    avg = 0
    l = 0

    for i in range(len(mom_avg)):
        avg = avg + mom_avg[i]
        l += 1

    mom_factor = float(avg / l) - mom_avg[0]
    return mom_factor

def cal_martingales():
    for i in u_list:
        mom_avg.append(calculate_avg_price_per_year(i))

    for i in range(len(mom_avg)):
        p = float(mom_avg[i])
        p = float(mom_avg[i] * mom_avg[i])
        s_mom_avg.append(p)



    sum_y = sum(s_mom_avg)
    sum_y = sum_y / len(mom_avg)
    sum_y = math.sqrt(sum_y)
    return sum_y

def cal_mean_rev():
    avg1 = calculate_avg()
    high_price = max(price)
    l = price.index(high_price)
    n = len(price)
    a = 0
    for i in range(l, n):
        if (price[i] == avg1 or price[i] < avg1):
            a = 1
    if (a == 1):
        return 0
    else:
        return 1

def cal_pe_ratio():
    for i in range(len(net_profit)):
        if (net_profit[i] == 0):
            earning.append(0)
        else:
            earning.append(float(net_profit[i] / volume[i]))

    for i in range(len(earning)):
        if (earning[i] < 0):
            l = float(c_price[i] / -1 * earning[i])
            l = -1 * l;
            p_e.append(float(c_price[i] / l))
        elif (earning[i] == 0):
            p_e.append(0)
        else:
            p_e.append(float(c_price[i] / earning[i]))

    if (p_e[-1] <= 150):
        flag = 1
    else:
        flag = 0

    return flag


X = []
target = []
path ="/home/rohit/Documents/thestocklist.csv"
file = open(path)
csvreader = csv.reader(file)
header = next(csvreader)
for row in csvreader:
    s = row[0]
    get_data(s)
    target.append(row[1])
    f=cal_momentum()
    g=cal_martingales()
    h=cal_mean_rev()
    j=cal_pe_ratio()
    s1 = [f,g,h,j]
    X.append(s1)
    clear_list()


n_hidden = 4
n_in = 4
n_out = 4
n_samples = 300

learning_rate = 0.01
momentum = 0.9

np.random.seed(0)

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def tanh_prime(x):
    return  1 - np.tanh(x)**2
def train(x, t, V, W, bv, bw):

    # forward
    A = np.dot(x, V) + bv
    Z = np.tanh(A)

    B = np.dot(Z, W) + bw
    Y = sigmoid(B)

    # backward
    Ew = Y - t
    Ev = tanh_prime(A) * np.dot(W, Ew)

    dW = np.outer(Z, Ew)
    dV = np.outer(x, Ev)

    loss = -np.mean ( t * np.log(Y) + (1 - t) * np.log(1 - Y) )

    return loss, (dV, dW, Ev, Ew)

def predict(x, V, W, bv, bw):
    sum_y = 0
    A = np.dot(x, V) + bv
    B = np.dot(np.tanh(A), W) + bw
    for i in range(4):
        sum_y = sum_y + B[i]
    return (sigmoid(sum_y) > 0.5).astype(int)

V = np.random.normal(scale=0.1, size=(n_in, n_hidden))
W = np.random.normal(scale=0.1, size=(n_hidden, n_out))

bv = np.zeros(n_hidden)
bw = np.zeros(n_out)

params = [V,W,bv,bw]

X = np.array(X).astype(dtype=np.uint8)
target = np.array(target).astype(dtype=np.uint8)
print(target)
T = target


for epoch in range(100):
    err = []
    upd = [0]*len(params)

    t0 = time.clock()
    for i in range(X.shape[0]):
        loss, grad = train(X[i], T[i], *params)

        for j in range(len(params)):
            params[j] -= upd[j]

        for j in range(len(params)):
            upd[j] = learning_rate * grad[j] + momentum * upd[j]

        err.append( loss )

    print ("Epoch: %d, Loss: %.8f, Time: %.4fs" % (
                epoch, np.mean( err ), time.clock()-t0 ))

x = [3.32,0,0,1]
print ("XOR prediction:")
print (x)
print (predict(x, *params))