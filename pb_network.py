# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 22:33:49 2017

@author: Administrator
"""

from numpy import *
import operator

def createDataSet():
    inputDataSet = array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]]).T
    ouputDataSet = array([[0,1,1,0]])
    return inputDataSet,ouputDataSet

    
def sigmoid(x):
    return 1/(1+math.exp(-x));

def runBpNetwork(inputData,w12,w23,theta2,theta3):
    layer1Dim = w12.shape[0];
    layer2Dim = w12.shape[1];
    layer3Dim = w23.shape[1];
    #print layer1Dim,layer2Dim,layer3Dim
    if layer2Dim <> w23.shape[0]:
        print('\n[ERROR] layer2Dim <> w23.shape[0]\n');

    if layer1Dim <> inputData.shape[0]:
        print('\n[ERROR] layer1Dim <> inputData.shape[0]\n');

    layer2Input = zeros(layer2Dim);
    layer2Ouput = zeros(layer2Dim);

    layer3Input = zeros(layer3Dim);
    layer3Ouput = zeros(layer3Dim);

    for i in range(layer2Dim):
        layer2Input[i] =  dot(inputData,w12[:,i]);
        layer2Ouput[i] =  sigmoid(layer2Input[i] - theta2[i]);

    for j in range(layer3Dim):
        layer3Input[j] =  dot(layer2Ouput,w23[:,j]);
        layer3Ouput[j] =  sigmoid(layer3Input[j] - theta3[j]);

    return layer2Ouput,layer3Ouput

#inputDataSet: the input  data set, which is arrayed as [x1;x2;x3;...;xN]
#ouputDataSet: the output data set, whith is arrayed as [y1;y2;y3;...;yN]
#w12    : the weight parameter between layer1 (input layer) and layer2 (intermediate layer)
#w23    : the weight parameter between layer2 (intermediate layer) and layer3 (putput layer)
#theta2 : the threshold parameter of layer2(intermediate layer)
#theta3 : the threshold parameter of layw3(output layer)
#maxIterNum : max iteration number that is allowed
#minLossRatio: min ratio of training error
#LeanRatio: the speed for the weight and threshold parameter learn from err
def trainBpNetwork(inputDataSet,ouputDataSet,intermediateLayerNum,maxIterNum,minLossRatio,LeanRatio):
    inputDataSetSize  = inputDataSet.shape[1];
    ouputDataSetSize  = ouputDataSet.shape[1];
    inputDataDim      = inputDataSet.shape[0];
    ouputDataDim      = ouputDataSet.shape[0];

    if inputDataSetSize <> ouputDataSetSize:
        print('\n[ERROR]inputDataSize is not equal to outputDataSize\n');

    w12    = random.normal(size=(inputDataDim,intermediateLayerNum));
    w23    = random.normal(size=(intermediateLayerNum,ouputDataDim));
    theta2 = random.normal(size=intermediateLayerNum);
    theta3 = random.normal(size=ouputDataDim);

#    print('\ninit parameter\n')
#    print w12
#    print w23
#    print theta2
#    print theta3

    iterIdx = 0;

    ouputTotalVal = trace(ouputDataSet * ouputDataSet.T);

    while iterIdx < maxIterNum:
        iterIdx = iterIdx + 1;
        totalLoss = 0;
        deltaTheta3 = zeros(ouputDataDim);
        deltaTheta2 = zeros(intermediateLayerNum);
        deltaW23    = zeros((intermediateLayerNum,ouputDataDim));
        deltaW12    = zeros((inputDataDim,intermediateLayerNum));

        for n in range(inputDataSetSize):
            inputData = inputDataSet[:,n];
            ouputData = ouputDataSet[:,n];
            layer2Ouput,layer3Ouput = runBpNetwork(inputData,w12,w23,theta2,theta3);
            errOutput = ouputData - layer3Ouput;
            loss = 0.5 * dot(errOutput,errOutput);
            totalLoss = totalLoss + loss;

            g = zeros(ouputDataDim);


            for j in range(ouputDataDim):
                g[j] = layer3Ouput[j] * (1 - layer3Ouput[j]) * (ouputData[j] - layer3Ouput[j]);
                deltaTheta3[j] = deltaTheta3[j] - LeanRatio * g[j];
                for h in range(intermediateLayerNum):
                    bh = layer2Ouput[h];
                    deltaW23[h][j] = deltaW23[h][j] + LeanRatio *  g[j] * bh;

            for h in range(intermediateLayerNum):
                bh = layer2Ouput[h];
                eh = bh * (1 - bh) * dot(w23[h,:],g);
                deltaTheta2[h] =  deltaTheta2[h] - LeanRatio * eh;
                for i in range(inputDataDim):
                    xi = inputData[i];
                    deltaW12[i][h] = deltaW12[i][h] + LeanRatio * eh * xi;

        w12 = w12 + deltaW12;
        w23 = w23 + deltaW23;
        theta2 = theta2 + deltaTheta2;
        theta3 = theta3 + deltaTheta3;

        lossRatio = totalLoss/ouputTotalVal;

        if lossRatio <= minLossRatio:
            break;

        if iterIdx > maxIterNum:
            break;


    

    return w12,w23,theta2,theta3,iterIdx,lossRatio

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
