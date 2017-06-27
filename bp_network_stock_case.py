# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 19:21:32 2017

@author: laizhenzhou@126.com
"""

from numpy import *

import tushare as ts   
import matplotlib.pyplot as plt
import pb_network

#调用中国联通的股票的历史数据。X是前30天的每日涨跌百分比，Y是后一天的涨跌逻辑值。
#trainSetLen ：训练样本集的长度
#testSetLen:   测试样本集的长度
#dateOffset:   测试样本集和训练样本集之间的间隔。 【训练样本时间分布】【dateOffset】【测试样本集时间分布】
#              如果dateOffset = 0,表示从训练样本集后面开始的第一天开始预测testSetLen天的涨跌情况。
def createStockDataSet1(trainSetLen, testSetLen, dateOffset):
    df = ts.get_hist_data('600050');
    #dateLen = df.shape[0];
    historyDateLen = 30;
    inputTrainDataSet = zeros((historyDateLen,trainSetLen));
    ouputTrainDataSet = zeros((1,trainSetLen));

    inputTestDataSet = zeros((historyDateLen,testSetLen));
    ouputTestDataSet = zeros((1,testSetLen));
    
    k = 0;
    for i in range(1+dateOffset+testSetLen,trainSetLen+dateOffset+testSetLen):
        inputTrainDataSet[:,k] = df['p_change'][i-1:i+historyDateLen-1];
        if df['p_change'][i] >= 0:
            ouputTrainDataSet[:,k] = 1;
        else:
            ouputTrainDataSet[:,k] = 0;
        k = k+1;
            
    k = 0;
    for i in range(1+dateOffset,testSetLen+dateOffset):
        inputTestDataSet[:,k] = df['p_change'][i-1:i+historyDateLen-1];
        if df['p_change'][i] >= 0:
            ouputTestDataSet[:,k] = 1;
        else:
            ouputTestDataSet[:,k] = 0;
        k = k+1;
    
    return inputTrainDataSet,ouputTrainDataSet,inputTestDataSet,ouputTestDataSet

#-----------生成测试数据集----------------
trainSetLen = 300
testSetLen  = 54
dateOffset  = 100
inputDataSet,ouputDataSet,inputTestDataSet,ouputTestDataSet = createStockDataSet1(trainSetLen,testSetLen,dateOffset)
print('\n输入训练数据集:')
print inputDataSet

print('\n输出训练数据集:')
print ouputDataSet

#-----------设置神经网络------------------
#隐层神经元个数
intermediateLayerNum = 20;
#最多迭代次数
maxIterNum           = 200;
#最小的迭代误差
minLossRatio         = 0.0001;
#学习速率系数 （0,1]
LeanRatio            = 0.2;

#-----------训练神经网络-------------------
w12,w23,theta2,theta3,iterIdx,lossRatio = trainBpNetwork(inputDataSet,ouputDataSet,intermediateLayerNum,maxIterNum,minLossRatio,LeanRatio);

# 给出最终的测试输入对应的预测输出
finalOutput = zeros(ouputDataSet.shape);
finalOutputLogic = zeros(ouputDataSet.shape);
for n in range(finalOutput.shape[1]):
    inputData = inputDataSet[:,n];
    ouputData = ouputDataSet[:,n];
    layer2Ouput,layer3Ouput = runBpNetwork(inputData,w12,w23,theta2,theta3);
    finalOutput[:,n] = layer3Ouput;



for i in range(finalOutput.shape[1]):
    if finalOutput[0,i] >= 0.5:
        finalOutputLogic[0,i] = 1;
    else:
        finalOutputLogic[0,i] = 0;
  
print('\n最终的训练集的涨跌预测输出')
print finalOutputLogic
    
errCnt = 0;
for i in range(finalOutput.shape[1]):
    if finalOutputLogic[0,i] != ouputDataSet[0,i]:
        errCnt = errCnt + 1;

print('\n最终训练集错误拟合次数，失误比率')
print errCnt,errCnt * 1.0/finalOutput.shape[1]

print('\n最终的训练集的拟合预测输出')
print finalOutput 


    
#print('\n最终12层之间的权重系数矩阵:')
#print w12
#print('\n最终23层之间的权重系数矩阵:')
#print w23
#print('\n最终第2层的偏置系数:')
#print theta2
#print('\n最终第3层的偏置系数:')
#print theta3
print('\n一共进行了的迭代次数:')
print iterIdx
print('\n拟合误差比率:')
print lossRatio

plt.figure()  #创建一个新的画布

plt.plot(ouputDataSet[0,:])
plt.hold
plt.plot(finalOutput[0,:])
plt.title(u'真实涨跌走势',fontproperties='SimHei')
plt.ylabel(u'涨跌',fontproperties='SimHei')
plt.xlabel(u'时间',fontproperties='SimHei')
plt.grid()
#-----------使用测试样本进行预测测试---------------------

predictTestOutput = zeros(ouputTestDataSet.shape);
predictTestOutputLogic = zeros(ouputTestDataSet.shape);
predictErrCnt = 0;

for n in range(testSetLen):
    inputData = inputTestDataSet[:,n];
    ouputData = ouputTestDataSet[:,n];
    layer2Ouput,layer3Ouput = runBpNetwork(inputData,w12,w23,theta2,theta3);
    predictTestOutput[:,n] = layer3Ouput;
    
    if predictTestOutput[0,n] >= 0.5:
        predictTestOutputLogic[0,n] = 1;
    else:
        predictTestOutputLogic[0,n] = 0;
    
    if predictTestOutputLogic[0,n] != ouputTestDataSet[0,n]:
        predictErrCnt = predictErrCnt + 1;
  
print('\n测试数据集（20天）的真实涨跌（0跌，1涨）:')
print ouputTestDataSet

print('\n测试数据集（20天）的预测涨跌（0跌，1涨）:')
print predictTestOutputLogic
    
print('\n错误预测次数，失误百分比')
print predictErrCnt,predictErrCnt*100.0/testSetLen

#print('\n最终的测试集的拟合预测输出')
#print predictTestOutput 
