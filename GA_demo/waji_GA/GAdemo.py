# -*- coding: utf-8 -*-
# @Time : 05/01/23 下午 02:36
# @Author : 张斌飞
# @Email : ZhangBinFei1995@outlook.com
# @File : GAdemo.py
# @Project : pythonProject1

import numpy as np
import matplotlib.pyplot as plt


class GA(object):
    '''
    在该求解最小值或最大值的过程中，可以自定义函数作为遗传算法的输入，分别设置  最大迭代次数，种群规模，基因长度，交叉概率，变异概率，
    自变量个数，自变量的变化范围，自定义函数值。
    通过改变get_fitness 的值来确定求函数的是最大值还是最小值。

    '''

    def __init__(self, N_GENERATIONS, population_size, chromosome_length, pc, pm, length, Parameter_BOUND, F,X=8,Z=6 ):
        self.N_GENERATIONS = N_GENERATIONS
        self.POP_SIZE = population_size
        self.DNA_SIZE = chromosome_length
        self.CROSSOVER_RATE = pc
        self.MUTATION_RATE = pm
        self.length = length
        self.Parameter_BOUND = Parameter_BOUND
        self.F = F
        self.X = X
        self.Z = Z

    # # 计算函数值：
    # def F(self, X):
    #     x = X[0]
    #     y = X[1]
    #     # z = X[2]
    #     return 3 * (1 - x) ** 2 * np.exp(-(x ** 2) - (y + 1) ** 2) - 10 * (x / 5 - x ** 3 - y ** 5) * np.exp(
    #         -x ** 2 - y ** 2) - 1 / 3 ** np.exp(-(x + 1) ** 2 - y ** 2)
    #     # return np.sin(x) + 1 + np.cos(x)
    #     # return 100.0 * (y - x ** 2.0) ** 2.0 + (1 - x) ** 2.0 + 4 * z

    # 计算函数适应度值
    def get_fitness(self, pop):
        X = self.translateDNA(pop)
        pred = self.F(X)
        # return (pred - np.min(pred)) + 1e-15  # (函数值越大，适应度值越大)
        # 减去最小的适应度是为了防止适应度出现负数，通过这一步fitness的范围为[0, np.max(pred)-np.min(pred)],最后在加上一个很小的数防止出现为0的适应度
        return (-(pred - np.max(pred)))**2 + 1e-15  # 求函数最小值时的适应度函数.(函数值越小，适应度值越大)


    def translateDNA(self, pop):  # pop表示种群矩阵，一行表示一个二进制编码表示的DNA，矩阵的行数为种群数目
        X = []
        for i in range(self.length):
            x_pop1 = pop[:, i:self.DNA_SIZE * self.length:self.length]  # 每隔几列取一个基因组成新的一个参数基因值
            x = x_pop1.dot(2 ** np.arange(self.DNA_SIZE)[::-1]) / float(2 ** self.DNA_SIZE - 1) * \
                (self.Parameter_BOUND[i][1] - self.Parameter_BOUND[i][0]) + self.Parameter_BOUND[i][0]
            X.append(x)

        return X

    def crossover_and_mutation(self, pop):
        ''' 这个地方进行了两个地方的交叉 '''
        new_pop = []
        for father in pop:  # 遍历种群中的每一个个体，将该个体作为父亲
            child = father  # 孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
            if np.random.rand() < self.CROSSOVER_RATE:  # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉
                mother = pop[np.random.randint(self.POP_SIZE)]  # 再种群中选择另一个个体，并将该个体作为母亲
                mother2 = pop[np.random.randint(self.POP_SIZE)]  # 再种群中选择另一个个体，并将该个体作为母亲
                cross_points = np.random.randint(low=0, high=self.DNA_SIZE)  # 随机产生交叉的点
                cross_points2 = np.random.randint(low=0, high=self.DNA_SIZE * self.length)  # 随机产生交叉的点
                child[cross_points:self.DNA_SIZE] = mother[cross_points:self.DNA_SIZE]  # 孩子得到位于交叉点后的母亲的基因
                child[cross_points2 :self.DNA_SIZE * self.length] = \
                    mother2[cross_points2:self.DNA_SIZE * self.length]

            self.mutation(child)  # 每个后代有一定的机率发生变异
            new_pop.append(child)

        return new_pop

    def mutation(self, child):
        ''' 这个地方我改变了 两个地方的基因 '''
        if np.random.rand() < self.MUTATION_RATE:  # 以MUTATION_RATE的概率进行变异
            mutate_point = np.random.randint(0, self.DNA_SIZE * self.length)  # 随机产生一个实数，代表要变异基因的位置
            mutate_point2 = np.random.randint(0, self.DNA_SIZE * self.length)  # 随机产生一个实数，代表要变异基因的位置
            child[mutate_point] = child[mutate_point] ^ 1  # 将变异点的二进制为反转
            child[mutate_point2] = child[mutate_point2] ^ 1  # 将变异点的二进制为反转

    def select(self, pop, fitness):  # nature selection wrt pop's fitness  自然选择WRT pop的适应性
        idx = np.random.choice(np.arange(self.POP_SIZE), size=self.POP_SIZE, replace=True,
                               p=(fitness) / (fitness.sum()))  # 适应度越高，被选择的机会越高，而适应度低的，被选择的机会就低。
        return pop[idx]

    def print_info(self, pop):
        fitness = self.get_fitness(pop)
        max_fitness_index = np.argmax(fitness)
        print("max_fitness:", fitness[max_fitness_index])
        X = self.translateDNA(pop)
        print("最优的基因型：", pop[max_fitness_index])
        X = np.array(X)
        print("此时的各个最优参数值为:", X[..., max_fitness_index])

        # print('此时的函数值为：', self.F(X[..., max_fitness_index]))
        print('此时的函数值为：', np.min(self.F(X)))

    def plot_fitness(self, Max_fitness, Average_fitness):
        plt.figure()
        plt.ion()
        plt.plot(range(self.N_GENERATIONS), Max_fitness, label='Max_fitness', color='red', linewidth=1.0,
                 linestyle='--')
        plt.plot(range(self.N_GENERATIONS), Average_fitness, label="Average_fitness")

        # df = pd.DataFrame(columns=['A', 'B'])
        # fig = df.plot(figsize=(10, 6))  # 创建图表对象，并复制给fig
        plt.title('GA')
        plt.xlabel('N_GENERATIONS ')
        plt.ylabel('fitness ')
        plt.legend(loc="upper right")  # 标签显示（一般称为图例）
        plt.ioff()
        plt.show()

    def plot_3d(self, x, y):
        # fig1 = plt.figure(figsize=(10, 6))
        # ax = Axes3D(fig1)   # 错误的语句

        plt.figure()
        plt.ion()
        ax = plt.axes(projection='3d')
        x = np.arange(-x, x, 0.1)
        y = np.arange(-y, y, 0.1)
        X, Y = np.meshgrid(x, y)
        Z = self.F([X, Y])
        plt.xlabel('x')
        plt.ylabel('y')
        ax.set_zlim([-5, 5])
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
        plt.title('Function image')
        plt.ioff()
        plt.show()

    def plot_2d(self, lb):
        plt.figure()
        plt.ion()
        x = np.linspace(-lb, lb)
        y = self.F([x])

        plt.plot(x, y)
        plt.ioff()
        plt.show()

    def main(self):

        pop = np.random.randint(2, size=(
            self.POP_SIZE, self.DNA_SIZE * self.length))  # 生成初始种群  matrix (POP_SIZE, DNA_SIZE*length)

        Average_fitness = []
        Max_fitness = []

        for i in range(self.N_GENERATIONS):  # 迭代N代

            pop = np.array(self.crossover_and_mutation(pop))  # 变异与交叉
            # F_values = F(translateDNA(pop)[0], translateDNA(pop)[1])#x, y --> Z matrix
            fitness = self.get_fitness(pop)
            pop = self.select(pop, fitness)  # 选择生成新的种群

            Average_fitness.append(np.mean(fitness))  # 每一代的平均是适应度
            Max_fitness.append(np.amin(fitness))  # 每一代的最大适应度值

            # # 计算每一代的最优函数值
            # Function_value = []
            # max_fitness_index = np.argmax(fitness)
            # X = self.translateDNA(pop)
            # x = X[0][max_fitness_index]
            # y = X[1][max_fitness_index]
            # z = X[2][max_fitness_index]
            # Function_value.append(self.F([x, y, z]))

        self.print_info(pop)
        self.plot_fitness(Max_fitness, Average_fitness)
        # 画函数图：
        if self.length == 2:
            p = np.array(self.Parameter_BOUND)
            self.plot_3d(10, 10)
        if self.length == 1:
            self.plot_2d(5)


if __name__ == '__main__':
    N_GENERATIONS = 100  # 迭代次数
    population_size = 50
    chromosome_length = 20
    pc = 0.8
    pm = 0.01



# # 一个参数时：
#     length = 1  # 参数数目，即需要求解的未知数个数
#     Parameter_BOUND = [-4, 4]
#     # 自定义函数内容
#     def F(X):
#         x = X[0]
#         return 10 * np.sin(x) + 10 * np.cos(x)
#
#
# # 2个参数时：
#     length = 2  # 参数数目，即需要求解的未知数个数
#     Parameter_BOUND = [[-4, 4],
#                         [-5 ,5]]
#
#     def F(X):
#         x= X[0]
#         y =X[1]
#         return 3 * (1 - x) ** 2 * np.exp(-(x ** 2) - (y + 1) ** 2) - 10 * (x / 5 - x ** 3 - y ** 5) * np.exp(
#             -x ** 2 - y ** 2) - 1 / 3 ** np.exp(-(x + 1) ** 2 - y ** 2)

    # 5个参数时：
    length =5 # 参数数目，即需要求解的未知数个数
    Parameter_BOUND = [[-4, 4],
                       [-5, 5],
                        [-4, 4],
                       [-5, 5],
                        [-4, 4],
                       ]
    def F(x):

        return x[0]**2+3*x[1]**2+x[3]+np.power(x[4],3)+np.power(x[2],2)


    ga = GA(N_GENERATIONS, population_size, chromosome_length, pc, pm, length, Parameter_BOUND, F)
    ga.main()
