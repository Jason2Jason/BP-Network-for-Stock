# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 18:50:14 2017

@author: laizhenzhou@126.com
"""
from numpy import *

def createDataSet():
    inputDataSet = array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]]).T
    ouputDataSet = array([[0,1,1,0]])
    return inputDataSet,ouputDataSet

#-----------生成测试数据集----------------
inputDataSet,ouputDataSet = createDataSet()
print('\n输入测试数据集:')
print inputDataSet

print('\n输出测试数据集:')
print ouputDataSet

#-----------设置神经网络------------------
#隐层神经元个数
intermediateLayerNum = 4;
#最多迭代次数
maxIterNum           = 2000;
#最小的迭代误差
minLossRatio         = 0.001;
#学习速率系数 （0,1]
LeanRatio            = 0.1;

#-----------训练神经网络-------------------
w12,w23,theta2,theta3,iterIdx,lossRatio = trainBpNetwork(inputDataSet,ouputDataSet,intermediateLayerNum,maxIterNum,minLossRatio,LeanRatio);

# 给出最终的测试输入对应的预测输出
finalOutput = zeros(ouputDataSet.shape);
for n in range(finalOutput.shape[1]):
    inputData = inputDataSet[:,n];
    ouputData = ouputDataSet[:,n];
    layer2Ouput,layer3Ouput = runBpNetwork(inputData,w12,w23,theta2,theta3);
    finalOutput[:,n] = layer3Ouput;
print('\n最终的测试集的预测输出')
print finalOutput
    
print('\n最终12层之间的权重系数矩阵:')
print w12
print('\n最终23层之间的权重系数矩阵:')
print w23
print('\n最终第2层的偏置系数:')
print theta2
print('\n最终第3层的偏置系数:')
print theta3
print('\n一共进行了的迭代次数:')
print iterIdx
print('\n拟合误差比率:')
print lossRatio

#-----------预测新样本---------------------
layer2Ouput,layer3Ouput = runBpNetwork(array([0,0,0.5]),w12,w23,theta2,theta3);
print('\n预测新样本[0,0,0.5]的输出:')
print layer3Ouput