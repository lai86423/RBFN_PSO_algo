import numpy as np
import math
import matplotlib.pyplot as plt

class plot_figure:
    #Initialize the track and car points
    def __init__(self):
        plt.close("all")
        self.degree = [0, 0, 90]
        self.final_point = [[18, 40], [30, 37]]
        self.point = [[-6, -3], [-6, 22], [18, 22], [18, 50], [30, 50],
         [30, 10], [6, 10], [6, -3], [-6, -3]]
        plot_figure.plot_map(self)
        plot_figure.plot_car(self, self.degree[0], self.degree[1], self.degree[2])

    #Draw the initial track and car 
    def plot_map(self):
       
        self.fig, self.ax = plt.subplots(1, 1, figsize=(4.2, 6.75))
        plt.xlim(-7, 32)
        plt.ylim(-5, 55)
       
        #Draw the Wall #各兩點逐一取出繪製成線段
        for i in range(len(self.point) - 1):
            self.ax.plot([self.point[i][0], self.point[i + 1][0]], [self.point[i][1], self.point[i + 1][1]], color = 'b')
        x = [self.final_point[0][0], self.final_point[1][0]]
        y = [self.final_point[0][1], self.final_point[1][1]]
        self.ax.plot(x, y, 'r')
        self.ax.plot([-7, 7], [0, 0], 'k')

    #Update and Draw the car
    def plot_car(self, pointX, pointY, phi):
        # 為避免一開始沒有車子可以消除，故用try
        global lines
        try:
            self.circle.remove()
            self.ax.lines.remove(lines[0]) 
        except Exception:
            pass
        #Draw the car
        self.circle = plt.Circle((pointX, pointY), 3, color='r', fill=False)
        self.ax.add_artist(self.circle)
        self.line_x = [4 * math.cos(phi * math.pi / 180) + pointX, pointX]
        self.line_y = [4 * math.sin(phi * math.pi / 180) + pointY, pointY]
        #Draw the car direction line 
        lines = plt.plot(self.line_x, self.line_y,'-r')

class Moving():
    def __init__(self, dim, node, weight):
        self.x = 0
        self.y = 0
        self.phi = 90
        self.canMove = True
        self.dim = dim
        self.weight = weight
        self.node = node
        self.point = np.loadtxt('case01.txt', delimiter = ',', skiprows=3)
        self.trackResult = []
        
    def main(self):
        self.detectWall = False
        self.detectFinish = False
        self.i = 0
        while(not any(self.DetectWall())):
            cloDist = self.sensor(self.point, self.x, self.y, self.phi)
            if self.dim == 4:
                data = cloDist
            else:
                data = [self.x, self.y, cloDist[0], cloDist[1], cloDist[2]]
            angle = self.RBFN_4D(data)
            self.x, self.y, self.phi = self.UpdataPos(angle)
            self.i += 1
            self.trackResult.append(self.x)
            self.trackResult.append(self.y)
            self.trackResult.append(self.phi)
        print('step', self.i)
        if self.detectFinish == True:
            print('success!!!!!!!')
            
        else:
            print('fail')

        return self.detectWall, self.detectFinish

    def UpdataPos(self, theta):
        
        x = self.x + math.cos((self.phi + theta) / 180 * math.pi) + math.sin(theta / 180 * math.pi) * math.sin(self.phi / 180 * math.pi)
        y = self.y + math.sin((self.phi + theta) / 180 * math.pi) - math.sin(theta / 180 * math.pi) * math.sin(self.phi / 180 * math.pi)
        phi = self.phi - (math.asin(2 * math.sin(theta / 180 * math.pi) / 6 )) * 180 / math.pi
        # print('pos', x, y, phi)

        return x, y, phi

    def DetectWall(self):
        
        if self.y <= 10:
            if(self.x >= (6 - 3) or self.x <= (-6 + 3) or self.y < (-3 + 3)):
                self.detectWall = True
                return self.detectWall, self.detectFinish
            else:
                self.detectWall = False
                return self.detectWall, self.detectFinish

        elif self.y <= 22-3:
            if(self.x <= (-6 + 3)or self.x >= (30 - 3)):
                self.detectWall = True
                return self.detectWall, self.detectFinish
            else:
                return self.detectWall, self.detectFinish
        elif self.y < 50:
            if(self.x <= (18 + 3) or self.x >= (30 - 3)):
                self.detectWall = True
                return self.detectWall, self.detectFinish
            elif self.y >= 37:
                self.detectFinish = True
                self.detectWall = False
                return self.detectWall, self.detectFinish
            else:
                return self.detectWall, self.detectFinish

    def sensor(self, point, pointX, pointY, phi):
        # 算出sensor的三條線個別點(圓心 -45度 0度 +45度)
        sensorX = [pointX, 4 * math.cos((phi + 45) * math.pi / 180) + pointX, 4 *
            math.cos(phi * math.pi / 180) + pointX, 
            4 * math.cos((phi - 45) * math.pi / 180) + pointX]
        sensorY = [pointY, 4 * math.sin((phi + 45) * math.pi / 180) + pointY, 4 * 
            math.sin(phi * math.pi / 180) + pointY, 
            4 * math.sin((phi - 45) * math.pi / 180) + pointY]

        min_a = [0, 0, 0]
        cloX = [0, 0, 0]
        cloY = [0, 0, 0]
        for i in range(len(point) - 1):
            # 取出待檢驗兩點的X與Y
            lineX1, lineY1, lineX2, lineY2 = point[i][0], point[i][1], point[i + 1][0], point[i + 1][1]
            # 為避免斜率為無意義，故加入微小的混淆項
            linDiffX = lineX1 - lineX2 + 0.00012
            linDiffY = lineY1 - lineY2 + 0.00013
            m = linDiffY / linDiffX

            # 將此線段分別與三個sensor計算長度
            for j in range(3):
                # 加上0.0001避免為0，最後四捨五入到小數點後兩位將不影響答案
                senDiffX = sensorX[j + 1] - sensorX[0] + 0.00011
                senDiffY = sensorY[j + 1] - sensorY[0] + 0.00012
                a = (lineY1 - pointY + pointX * m - lineX1 * m) / (senDiffY - senDiffX * m)
                # 若向量的倍數a非大於零，代表沿著sensor出去的線是向後的，故不考慮
                if a > 0:
                    # 確保a的值與前面線段比較過後，確實為較小的值
                    if a <= min_a[j] or min_a[j] == 0:
                        # 交點x y
                        temp_x = a * senDiffX + pointX
                        temp_y = a * senDiffY + pointY
                        # 確保交點是在線段範圍內，而非線段的延伸
                        if (lineX1 - 0.1 <= temp_x <= lineX2 + 0.1) or (lineX2 - 0.1 <= 
                                temp_x <= lineX1 + 0.1):
                            if (lineY1 - 0.1 <= temp_y <= lineY2 + 0.1) or (lineY2 - 0.1 <=
                                    temp_y <= lineY1 + 0.1):
                                min_a[j] = a
                                cloX[j] = round(temp_x * 100) / 100
                                cloY[j] = round(temp_y * 100) / 100
        cloDist = [0, 0, 0]
        # 利用算出最近的三個點，計算到車中心的距離
        for i in range(3):
            cloDist[i] = Moving.distance(pointX, pointY, cloX[i], cloY[i])
        cloDist[0], cloDist[1], cloDist[2] = cloDist[1], cloDist[2], cloDist[0]
        return cloDist

    def distance(x1, y1, x2, y2):
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** (1/2)

    def RBFN_4D(self, data):
        for i in range(len(data)):
            data[i] = (data[i] / 80) * 2 - 1
        baseFunc = np.zeros(self.node)
        for j in range(self.node): 
            temp = 0
            for k in range(len(data)):
                temp += ((data[k] - self.weight[1 + self.node + j * k + k]) ** 2) #該nodej對 Xik 各維計算（Xik - mkj) ^ 2 後相加    
            baseFunc[j] = math.exp(temp / ((-2) * (self.weight[j - self.node] ** 2))) #再除該nodej對應 σ ^ 2 * (-2)
        objFunc = np.sum(baseFunc * self.weight[1 : 1 + self.node]) + self.weight[0] #F(x)= sum w*σ + theta    
        angle = math.tanh(objFunc) * 40

        return angle


