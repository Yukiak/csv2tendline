#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from decimal import Decimal,ROUND_HALF_UP
from tqdm import tqdm

"""RMS算出"""

def RMS(x,y):
        
    rms=np.zeros((len(x),1))
    
    #RMS_NSAT
    if len(x.columns)==1 and len(y.columns)==1:
      X="NSAT"
      print("RMS算出中_NSAT")

      param_max=int(input("param_maxを入力してください(ex/param_max=32,stepは1で固定)"))
      residual_max=int(input("residual_maxを設定しますか？(No=0,Yes=設定したいresidual_maxの値)"))

      step=1
      for param in tqdm(np.arange(0,param_max,step)):
        k=0
        l=0
        for i in range(len(x)):
          if pd.isna(y.iloc[i,0])==False and x.iloc[i,0]==param: 
              k+=1
              l+=pow(y.iloc[i,0],2)
              rms[int(Decimal(param/step).quantize(Decimal("0"),rounding=ROUND_HALF_UP)),0]=np.sqrt(l/k)
          elif k==0:
              rms[int(Decimal(param/step).quantize(Decimal("0"),rounding=ROUND_HALF_UP)),0]=0

    #RMS_DOP
    elif len(x.columns)==5 and len(y.columns)==1:
      print("RMS算出中_DOP")

      DOP=int(input("DOPを選択してください(G=0,P=1,H=2,V=3,T=4)"))
      print("param_maxとstepを入力してください(ex/param_max=4.5,step=0.01)")
      param_max,step=map(float,input().split())

      DOP_type=["G","P","H","V","T"]
        
      X="{}DOP".format(DOP_type[DOP])

      residual_max=int(input("residual_maxを設定しますか？(No=0,Yes=設定したいresidual_maxの値)"))

      for param in tqdm(np.arange(0,param_max,step)):
        k=0
        l=0
        for i in range(len(x)):
          if pd.isna(y.iloc[i,0])==False and abs(Decimal(x.iloc[i,DOP])-Decimal(str(param)))<1e-5:
              k+=1
              l+=pow(y.iloc[i,0],2)
              rms[int(Decimal(param/step).quantize(Decimal("0"),rounding=ROUND_HALF_UP)),0]=np.sqrt(l/k)
          elif k==0:
              rms[int(Decimal(param/step).quantize(Decimal("0"),rounding=ROUND_HALF_UP)),0]=0

    #RMS_EL
    else:
      X="EL"
      print("RMS算出中_EL")

      step=int(input("stepを入力してください(ex/step=5,param_maxは91で固定)"))

      residual_max=int(input("residual_maxを設定しますか？(No=0,Yes=設定したいresidual_maxの値)"))

      param_max=91
      for param in tqdm(range(0,param_max,step)):
        k=0
        l=0
        for i in range(len(x)):
          for j in range(len(x.columns)):
            if pd.isna(y.iloc[i,j])==False and x.iloc[i,j]==param:
                k+=1
                l+=pow(y.iloc[i,j],2)
                rms[int(Decimal(param/step).quantize(Decimal("0"),rounding=ROUND_HALF_UP)),0]=np.sqrt(l/k)
            elif k==0:
                rms[int(Decimal(param/step).quantize(Decimal("0"),rounding=ROUND_HALF_UP)),0]=0
    
    #RMSが出力される最終行+1を求める
    init=0
    if param_max%step==0:
      init=int(Decimal(param/step).quantize(Decimal("0"),rounding=ROUND_HALF_UP))+1
    else:
      init=int(Decimal(np.ceil(param_max/step)).quantize(Decimal("0"),rounding=ROUND_HALF_UP))

    #データ整理
    rms=np.delete(rms,range(init,len(x),1),axis=0)
    x_pre=pd.DataFrame(np.arange(0,param_max,step))
    y_pre=pd.DataFrame(rms)
    
    #結果出力用のDataFrame
    df4=pd.concat([x_pre,y_pre],axis=1)
    df3_prime={}
    if X=="EL":
      for i in range(len(x.columns)):
        df3_prime[i]=pd.DataFrame(index=range(len(x)),columns=["x","y"])
        for j in range(len(x.columns)):
          df3_prime[j]=pd.concat([x.iloc[:,j],y.iloc[:,j]],axis=1)
          df3_prime[j].columns=["x","y"]
          if j!=0:
            df3_prime[0]=pd.concat([df3_prime[0],df3_prime[j]])
    elif X=="DOP":
      df3_prime[0]=pd.DataFrame(index=range(len(x)),columns=["x","y"])
      df3_prime[0]=pd.concat([x.iloc[:,DOP],y.iloc[:,0]],axis=1)
      df3_prime[0].columns=["x","y"]
    else:
      df3_prime[0]=pd.DataFrame(index=range(len(x)),columns=["x","y"])
      df3_prime[0]=pd.concat([x.iloc[:,0],y.iloc[:,0]],axis=1)
      df3_prime[0].columns=["x","y"]
    
    df3_5=df3_prime[0]
    
    df3_5.dropna(how="any",inplace=True)

    df3_5.reset_index(inplace=True,drop=True)
    
    if X!="DOP":
      df3_5["x"]=df3_5["x"].astype("int")
    
    for i in range(len(df3_5)-1,-1,-1):
      if residual_max!=0 and df3_5.iloc[i,1]>residual_max:
        df3_5.drop(index=i,inplace=True)
    
    #residual_maxの設定
    for param in np.arange(0,param_max,step)[::-1]:
      if residual_max==0:
         if y_pre.iloc[int(Decimal(param/step).quantize(Decimal("0"),rounding=ROUND_HALF_UP))][0]==0:
            df4=df4.drop(index=int(Decimal(param/step).quantize(Decimal("0"),rounding=ROUND_HALF_UP)))
      else:
         if y_pre.iloc[int(Decimal(param/step).quantize(Decimal("0"),rounding=ROUND_HALF_UP))][0]==0 or             y_pre.iloc[int(Decimal(param/step).quantize(Decimal("0"),rounding=ROUND_HALF_UP))][0]>residual_max:
            df4=df4.drop(index=int(Decimal(param/step).quantize(Decimal("0"),rounding=ROUND_HALF_UP)))
    
    df4.columns=["{}".format(X),"疑似距離残差RMS"]
    
    #グラフ描画
    fig1=plt.figure()
    
    plt.title("range error model")
    plt.xlabel("{}".format(X))
    plt.ylabel("pseudo-range residual")
    plt.grid(True)
    
    if len(x.columns)==len(y.columns):
       plt.scatter(x,y)
    else:
       plt.scatter(x.iloc[:,DOP],y)
    
    return (rms,df3_5,df4,param_max,step,residual_max,X,fig1)

