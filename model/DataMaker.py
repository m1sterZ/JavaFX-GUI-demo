import  torch
import  torch.nn as nn
import  numpy as np
import  random
class dataMaker():
    def __init__(self):
       self.input=[]
       self.output=[]
       self.map={1:'+',2:'-',3:'*',4:'/'}

    def caculate(self,a,b,operator):
        result="null"
        if operator=='+':
            result=a+b
        if operator == '-':
            result = a-b
        if operator == '*':
            result = a*b
        if operator == '/':
            result = a/b
        return result

    def interpreter(self, sample):
        operator1=self.map[sample[1]]
       # operator2=self.map[sample[3]]
       # temp1=0
       # if operator1=='*' or operator1=='/':
         #   temp1= self.caculate(sample[0],sample[2],operator1)
       #     return  self.caculate(temp1,sample[4],operator2)
        #elif operator2=='*' or operator2=='/':
       #     temp1 = self.caculate(sample[2], sample[4], operator2)
        #    return self.caculate(sample[0],temp1,  operator1)
      #  else:
            #temp1=self.caculate(sample[0],sample[2],operator1)
           # return self.caculate(temp1, sample[4], operator2)
        return self.caculate(sample[0], sample[2], operator1)
    def InputMaker(self,runtimes):
        for i in range(runtimes):
            operator1=random.randint(1,1000)
            #operator2=random.randint(1,4) # 1,2,3,4对应+，-，*，/
            operator2=1
            operator3=random.randint(1,1000)
            sample = [operator1, operator2, operator3 ]
         #   operator4=random.randint(1,4)
          #  operator5=random.uniform(1,1000)
           # sample=[operator1,operator2,operator3,operator4,operator5]
            self.input.append(sample)

    def OutMaker(self):
        for i in range(len(self.input)):
            sample=self.input[i]
            self.output.append(self.interpreter(sample))

# maker=dataMaker()
# maker.InputMaker(100000)
# maker.OutMaker()
# np.save('Input.npy',maker.input)
# np.save('Output.npy',maker.output)
# print(maker.input[1])
# print(maker.output[1])
Input=[]
Output=[]
for i in range(1000000):
    a=random.uniform(0, 100)
    b=random.uniform(0,100)
    c=random.randint(1,100)
    d = random.randint(1, 100)
    e=random.randint(1, 100)
    f=random.randint(1, 100)
    Input.append([a,b,c,d,e,f])
    temp=0
    temp2=0
    temp3=0
    temp4=0
    for j in range(1):
        temp=a*b+3*temp
        a=a*0.98+0.2*b
        temp2=b*2+0.5*a
        temp2=0.5*c+temp*0.2*d
        temp3=2*a+3*b+temp
        temp3=temp3+c*d
        temp4=temp4+e*0.3*f
        temp4=temp4*0.2*f
    Output.append([temp,temp2,temp3,temp4])
np.save('Input.npy', Input)
np.save('Output.npy', Output)