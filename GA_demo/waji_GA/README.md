


这个可以调用。调用方法为：


from GA_demo.GAdemo import GA
import numpy as np
N_GENERATIONS = 100  # 迭代次数
population_size = 50
chromosome_length = 20
pc = 0.6
pm = 0.01
length = 1  # 参数数目，即需要求解的未知数个数
Parameter_BOUND = [[-4, 4],
                   [2, 8],
                   [9, 10]
                   ]


# 自定义函数内容
def F(X):
    x = X[0]
    # y = X[1]
    # z = X[2]
    # return 3 * (1 - x) ** 2 * np.exp(-(x ** 2) - (y + 1) ** 2) - 10 * (x / 5 - x ** 3 - y ** 5) * np.exp(
    #     -x ** 2 - y ** 2) - 1 / 3 ** np.exp(-(x + 1) ** 2 - y ** 2)
    return np.sin(x)


ga = GA(N_GENERATIONS, population_size, chromosome_length, pc, pm, length, Parameter_BOUND, F)
ga.main()



