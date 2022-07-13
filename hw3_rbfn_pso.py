import numpy as np 
import math
import matplotlib.pyplot as plt
import time
from trackTest import Moving
from trackTest import plot_figure
import tkinter as tk
import tkinter.ttk as ttk

def loadData(fileName): 
    data = np.loadtxt(fileName)
    dim = len(data[0])
    for i in range(len(data)):
        for n in range (dim - 1):
            data[i][n] = ( data[i][n] / 80 ) * 2 - 1    # 四維資料前3項:距離 值域0~80 正規化到-1~1
        data[i][-1] = ( data[i][-1] / 40 )  # 最後一項:角度 值域-40~40 正規化到-1~1
    
    return data

#----------------------------------------------------------------------------
# 生成基因向量
def initWeight(dim_i, particle, node_j):
    genDim = 1 + node_j + (dim_i * node_j) + node_j     # 基因向量維度
    weight1 = np.random.rand(particle, genDim - node_j)* 2 - 1     # theta,w,m 值域-1~1
    weight2 = np.random.rand(particle, node_j)     # sigma(σ) 值域0~1(Gussian)
    weight = np.hstack((weight1, weight2)) 
    return weight

#----------------------------------------------------------------------------
# RBFN求適應函數
def RBFN(data, weight, particle, node, dim):
    fitFunc = np.zeros(particle)   # 適應函數值有族群數個
    baseFunc = np.zeros(node)
    result = []     # TODO 取出最好的結果，供測試軌道與比較loss function
    for g in range (particle): 
        objTemp = 0
        for i in range(len(data)):  # 第i筆x
            for j in range(node): 
                temp = 0
                for k in range(dim):
                    temp += (data[i][k] - weight[g][1 + node + j * k + k]) ** 2 #該nodej對 Xik 各維計算（Xik - mkj) ^ 2 後相加   
                baseFunc[j] = math.exp(temp / ((-2) * ((weight[g][j - node]) ** 2))) #再除該nodej對應 σ ^ 2 * (-2)
            objFunc = np.sum(baseFunc * weight[g][1 : 1 + node]) + weight[g][0] #F(x)= sum w*σ + theta
            
            result.append(math.tanh(objFunc))   # 將所有particle的適應函數存起來
            objTemp += (data[i][-1] - math.tanh(objFunc)) ** 2      # tanh解決超過1 -1 的問題
        objTemp /= 2    # E(n) = 1/2*sum(yn-F(x))uj
        fitFunc[g] = 1 / objTemp    # 適應函數越小越佳，於是求倒數
    obj = 1 / fitFunc
    minIndex = np.argmin(obj)   # 取出最低的適應函數值
    result = result[minIndex * len(data) : (minIndex + 1) * len(data)] 
    
    for i in range(len(result)):
        result[i] = result[i] * 40

    return fitFunc, result, minIndex

#----------------------------------------------------------------------------
# 實數型基因演算法
def geneAlgo(fitFuncO, phi_one, phi_twoP, weight, node, particle, minIndex): #實數型基因演算法 
    #-------------------------------------
    # Reproduction  (因已算出適應函數數值，複製採用輪盤式選擇)   
    bestWei = weight[minIndex]      # 取出最好的weight
    fitFunc = np.copy(fitFuncO) 
    fitSum = np.sum(fitFunc) 
    for i in range (len(fitFunc)):
        fitFunc[i] = np.around((fitFunc[i] / fitSum) * particle)   # 求該基因在全部佔的比例 * 總數 = 被複製數
    fitRank = np.argsort(fitFunc)   # argsort求排序的對應index（小到大）
    
    ## 處理依比例四捨五入算基因數量，會有比原欲取族群多或少一點(e個)的問題
    e = np.sum(fitFunc) - particle 
    count = 0
    if e > 0:   # 將最小“非零”的e個基因扣掉
        for i in range(len(fitRank)):
            if fitFunc[fitRank[i]] == 0:    # 最小但為零不取，因為在fitFunc沒佔任何比例個
                pass
            else:
                fitFunc[fitRank[i]] -= 1 
                count += 1
                if count == e:
                    break           
    elif e < 0:     #   將最大的e個基因補足
        for i in range(int(-e)):
            fitFunc[fitRank[-i]] += 1
    
    ## 計算複製後對應之基因向量weight
    for j in range(particle):
        if fitFunc[j] != 0 :
            for i in range(int(fitFunc[j])):     # 幾個基因
                try:
                    newWeight = np.append(newWeight, weight[j], axis=0)               
                except:     # newWeight還未定義時跑這裡
                    newWeight = weight[j]            
    newWeight = np.reshape(newWeight, (particle, -1))  # 重整矩陣維度為 particle ＊ genDim
    
    #---------------------------------------
    #--phi_two-- 
    crossNum = int(phi_twoP * particle)
    if crossNum % 2 == 1:    # 交配數量必須為偶數
        crossNum += 1
    np.random.shuffle(newWeight)    
    fuzzyP = 0.4
    for i in range(0, crossNum, 2):     # 每次取兩個交配
        fuzzy = fuzzyP * (newWeight[i] - newWeight[i+1])
        # ---實數型交配公式： x1' = x1 + σ(x2 - x1) # x2' = x1 - σ(x2 - x1)
        newWeight[i], newWeight[i+1] = newWeight[i] + fuzzy ,newWeight[i+1] - fuzzy
    #-------------------------------------
    #--Mutation--
    MutaNum = int(phi_one * particle)
    a = np.random.rand(MutaNum, len(newWeight[0]))* 2 - 1 
    #為了使後面newWeight += a加雜訊的突變越來越大 *2-1 使值域變成-1~1 有加有減 
    
    a[a > phi_one] = 0    # 隨機取後大於突變機率的“位置” 的值改為0
    a[a < -phi_one] = 0
    a *= 0.2    # 讓值變小
    b = np.zeros((particle - MutaNum, len(newWeight[0])))
    a = np.vstack((a, b))   # 把維度補成跟weight一樣 才能和weight運算
    newWeight += a
    newWeight[-1] = bestWei     # 最後一項更改為最好的weight

    return newWeight, bestWei

#----------------------------------------------------------------------------
# SaveModelParams
# Θ
# 第一顆神經元參數: W_1  m_11  m_12  m_13  σ_1
# 第二顆神經元參數: W_2  m_21  m_22  m_23  σ_2
def SaveModelParams(weight,node,particle,dim):
    theta = weight[0]
    w = weight[1:1 + node]
    m = weight[1 + node:-node]
    sigma = weight[-node:]
    f = open("RBFN_params.txt",'w')
    f.writelines(str(theta) + '\n')
    for i in range(node):
        f.writelines(str(w[i]) + ' ')
        for d in range(dim): 
            f.writelines(str(m[dim * i + d]) + ' ')
        f.writelines(str(sigma[i]) + '\n')
    f.close()   

#----------------------------------------------------------------------------
# LoadModelParams
def LoadModelParams(file):  
    theta = np.loadtxt(file, max_rows = 1)
    theta = np.reshape(theta,(1,-1))
    geneVec = np.loadtxt(file, delimiter =" ", skiprows = 1)
    geneVec = np.split(geneVec,[1, -1],axis = 1)
    w = geneVec[0].T
    m = np.reshape(geneVec[1].T,(1,-1),order='F')
    sigma = geneVec[2].T
    weight = np.hstack((theta, w, m, sigma))[0]
    #print("w.shape",w.shape[1])
    node = w.shape[1]
    Xdim = m.shape[1] / node

    return weight, node, Xdim + 1  #！！dim 統一為輸入資料維度！！

def plot_track(trackResult):
    fig = plot_figure()
    plt.ion()
    plt.show()
    for i in range(0, len(trackResult), 3):
        x, y, phi = trackResult[i], trackResult[i + 1], trackResult[i + 2]
        fig.plot_car(x, y, phi)
        plt.show()
        plt.pause(0.05)
    plt.ioff()

def loadPar():
    print('loadPar')
    fileName = 'RBFN_params.txt'
    bestWei, node, dim = LoadModelParams(fileName)
    print(dim, node)
    print(bestWei)
    finalCar = Moving(dim, node, bestWei)
    detectWall, detectFinish = finalCar.main()
    trackResult = finalCar.trackResult
    plt.figure(3)
    plot_track(trackResult)

#----------------------------------------------------------------------------
#PSO Algo

def PSOAlgo(velocity, selfFic, selfwei, weight, fitFunc, phi_one, phi_twoP, minIndex, particle):
    bestWei = weight[minIndex]
    maxVelocity = 0.4
    for i in range(particle):
    
        velocity[i] = velocity[i] + phi_one * (selfwei[i] - weight[i]) + phi_twoP * (bestWei - weight[i])
        velocity[i][velocity[i] > maxVelocity] = maxVelocity
        velocity[i][velocity[i] < -maxVelocity] = -maxVelocity
        weight[i] = weight[i] + velocity[i]

    return weight, velocity, bestWei
#----------------------------------------------------------------------------

def Main():
    fileName = str(var_filename.get()) 
    if fileName ==  "train4dAll.txt" or fileName =='train4D.txt': 
        dim = 4
    else:
        dim = 6

    #Get input Variable
    data = loadData(fileName)
    epoch = int(var_epoch.get())
    particle = int(var_particle.get())
    node = int(var_node.get())
    phi_one = float(var_phi_one.get())
    phi_twoP = float(var_phi_twoP.get())
    optin_saveM = var_option.get() #0/1

    #initWeight 
    weight = initWeight(dim - 1, particle, node)   #x_i => dim - 1：扣掉最後一項預期輸出
    y = []
    print("weight[0].shape",weight[0].shape,len(weight[0]))
    velocity =  np.zeros((particle,len(weight[0])))
    
    start_time = time.time()
    #Training by RBFN to get Fitness fuction and get best weight by PSO Algo 
    for i in range (epoch):
        print('epoch {0} / {1}'.format(i, epoch))
        fitFunc, result, minIndex = RBFN(data, weight, particle, node, dim - 1)
        try:
            for g in range(particle):
                if selfFic[g] > fitFunc[g] :
                    selfFic[g] = fitFunc[g]
                    selfwei[g] = weight[g]
        except:
            selfFic = fitFunc
            selfwei = weight
        
        weight, velocity, bestWei = PSOAlgo(velocity, selfFic, selfwei, weight, fitFunc, phi_one, phi_twoP, minIndex, particle)
        #weight, bestWei = geneAlgo(fitFunc, phi_one, phi_twoP, weight, node, particle, minIndex)
        y.append(min(1 / fitFunc))
        print('fitFunc', min(1 / fitFunc))
        
        #If get the best weight,try to let the car run on the track 
        try:
            if not(lastBestWei == bestWei).all():
                car = Moving(dim, node, bestWei)
                detectWall, detectFinish = car.main()
                lastBestWei = np.copy(bestWei)
        except:
            lastBestWei = np.copy(bestWei)
            detectFinish = False
        
        print('-' * 30)
        if detectFinish == True:
            end_time = time.time()
            print(lastBestWei)
            print("PSOAlgo Running Time = ",end_time - start_time)
            if optin_saveM == 0 :
                SaveModelParams(bestWei, node, particle, dim - 1)
            trackResult = car.trackResult
            plt.figure(1)
            plot_track(trackResult)
            break
    
    plt.figure(2)
    plt1 = plt.subplot(211)     # loss function
    plt2 = plt.subplot(212)     # 期望輸出與實際輸出

    b = np.arange(data.shape[0])

    plt2.plot(b, data[:, -1] * 40, label='training_data')

    x = np.arange(len(y))
    plt1.plot(x, y)
    plt1.set_title('loss_function')

    b = np.arange(len(result))
    plt2.plot(b, result, label='real_output')
    plt2.set_title('output')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    #callback Funtion
    def radbut_click():
        selected_item = var_option.get()
        label_result.config(text=saveOption[selected_item][0])
    #----------------------------------------------------------------------------
    #GUI  #迭代次數 族群大小 突變機率 交配機率 選擇訓練資料集 Save/Load model params
    window = tk.Tk()
    window.geometry('430x330')
    window.title('HW3 - RBFN x PSO Algo')

    frame = tk.Frame()
    frame.pack(side=tk.TOP, pady=5)

    #檔案選擇設定
    file_option=('train4D.txt','train6D.txt','train4dAll.txt','train6dAll.txt')
    var_filename=tk.StringVar()
        
    ##下拉選單設定
    combobox=ttk.Combobox(window,values=file_option,textvariable=var_filename)
    combobox.pack(side=tk.TOP, pady=5)

    frame1 = tk.Frame()
    frame1.pack(side=tk.TOP, pady=5)

    label_epoch = tk.Label(frame1, text='Epoch = ')
    label_epoch.pack(side=tk.LEFT, padx=(10,0))

    var_epoch=tk.StringVar()
    ent_epoch = tk.Entry(frame1, textvariable=var_epoch, width=4)
    ent_epoch.pack(side=tk.LEFT)

    label_particle = tk.Label(frame1, text='Particle Num= ')
    label_particle.pack(side=tk.LEFT, padx=(10,0))

    var_particle=tk.StringVar()
    ent_particle = tk.Entry(frame1, textvariable = var_particle, width=4)
    ent_particle.pack(side=tk.LEFT)

    label_node = tk.Label(frame1, text='RBFN Node = ')
    label_node.pack(side=tk.LEFT, padx=(10,0))

    var_node=tk.StringVar()
    ent_node = tk.Entry(frame1, textvariable = var_node, width=4)
    ent_node.pack(side=tk.LEFT)

    frame2 = tk.Frame()
    frame2.pack(side=tk.TOP, pady=5)

    label_phi_one = tk.Label(frame2, text='φ1 = ')
    label_phi_one.pack(side=tk.LEFT, padx=(10,0))

    var_phi_one=tk.StringVar()
    ent_phi_one = tk.Entry(frame2, textvariable=var_phi_one, width=5)
    ent_phi_one.pack(side=tk.LEFT, pady=5)

    label_phi_twoP = tk.Label(frame2, text='φ2 = ')
    label_phi_twoP.pack(side=tk.LEFT, padx=(10,0))

    var_phi_twoP=tk.StringVar()
    ent_phi_twoP = tk.Entry(frame2, textvariable = var_phi_twoP, width=5)
    ent_phi_twoP.pack(side=tk.LEFT, pady=5)

    frame3 = tk.Frame()
    frame3.pack(side=tk.TOP, pady=5)
    
    label_saveModel = tk.Label(frame3, text='Save Model Params ? ')
    label_saveModel.pack(side=tk.LEFT, padx=(10,0))

    saveOption = (('Yes',0),('No',1))
    var_option = tk.IntVar()
    var_option.set(0)

    for item, value in saveOption:
        radbut = tk.Radiobutton(frame3, text=item, variable=var_option,
         value=value, command=radbut_click)
        radbut.pack(side=tk.LEFT, pady=5)

    label_result = tk.Label(frame3, fg='orange red')
    label_result.pack(side=tk.LEFT, pady=5)

    frame4 = tk.Frame()
    frame4.pack(side=tk.TOP, pady=5)

    button_start = tk.Button(frame4, text='Start Training', command=Main)
    button_start.pack(side=tk.TOP)

    frame5 = tk.Frame(bg='AntiqueWhite2')
    frame5.pack(side=tk.TOP, pady=30, fill=tk.BOTH)
    
    label_loadModel = tk.Label(frame5, text='Load Saved Model Params',bg='AntiqueWhite2')
    label_loadModel.pack(side=tk.TOP, padx=(10,0))

    button_load = tk.Button(frame5, text='Load', command=Main)
    button_load.pack(side=tk.TOP)

    window.mainloop()

