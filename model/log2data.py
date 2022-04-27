# encoding: utf-8
import os
from pathlib import Path
import linecache
import re
import sys

# 所有路径使用绝对路径

def readlog(filename):
    global dirname, lineNumSet
    i = 1
    linetext = linecache.getline(filename, i)
    methodNameSet = set()
    lineNumSet = set()
    if (linetext != ''):
        elems = linetext.split("@@|")
        # if (elems[0] != None):
        #     dirname = './' + elems[0]
        #     os.mkdir(dirname)
        tmp = filename.split('\\')
        last = tmp[len(tmp) - 1]
        dirname = filename.replace(last, '')
        # print(dirname)
    while linetext != '':
        if i not in lineNumSet:
            methodNameSet, lineNumSet = recordMethodData(dirname, filename, methodNameSet, lineNumSet, i)
        i = i + 1
        linetext = linecache.getline(filename, i)
    getIOdata(dirname)


def recordMethodData(dirname, filename, methodNameSet, lineNumSet, i):
    linetext = linecache.getline(filename, i)
    if linetext != '':
        elems = linetext.split("@@|")
        if len(elems) < 12:
            print(linetext)
            return methodNameSet,lineNumSet
        else:
            methodName = elems[3] + '.' + elems[4] + '.' + elems[5]

        if methodName not in methodNameSet and elems[2] == "COMMON":
            # startRecord
            lineNumSet = recordImpl(dirname, filename, methodName, lineNumSet, i)
            methodNameSet.add(methodName)

    return methodNameSet, lineNumSet

def recordImpl(dirname, filename, methodName, lineNumSet, i):
    linetext = linecache.getline(filename, i)

    while linetext != '':
        elems = linetext.split("@@|")
        if len(elems) < 12:
            return lineNumSet
        else:
            this_methodName = elems[3] + '.' + elems[4] + '.' + elems[5]

        if this_methodName == methodName:
            this_tid = elems[8]
            # 按照线程号写入文件
            write_data(dirname, methodName, this_tid, elems)
            lineNumSet.add(i)

        i = i + 1
        linetext = linecache.getline(filename, i)

    return lineNumSet

# 缓存
def write_data(dirname, methodName, this_tid, elems):
    cachedir = Path(dirname + '\\cache')
    if not cachedir.exists():
        os.mkdir(cachedir)
    filename = dirname + '\\cache\\' + methodName + '_' + this_tid
    file = open(filename, 'a')
    file.write(elems[6] + '@@|' + elems[9] + '\n')

def getIOdata(dirname):
    datadir = dirname
    # os.mkdir(datadir)

    for filename in os.listdir(dirname + 'cache\\'):
        # if not os.path.isdir(os.path.join(dirname + './cache/', filename)):  # 去掉的话会有文件夹的目录被打印
        # print('getIOdata', filename)
        writeIOdata(dirname, filename, datadir)

def writeIOdata(dirname, filename, datadir):
    logname = dirname + 'cache\\' + filename
    i = 1
    stack = []
    logevent = linecache.getline(logname, i)
    # 创建新文件存放输入输出数据
    IOdataname = datadir + filename.split('_')[0]+'.txt'
    print(IOdataname)
    file = open(IOdataname, 'w')
    while logevent != '':
        eventType = logevent.split('@@|')[0]
        data = logevent.split('@@|')[1] 
        if eventType == 'START':
            stack.append(i)
        else:
            inline = stack.pop()
            Input = getInput(linecache.getline(logname, inline).split('@@|')[1].strip('\n'))
            Output = getOutput(data.strip('\n'))
            # 把输入输出写入到文件中
            file.write(Input + '|' + Output + '\n')

        i = i + 1
        logevent = linecache.getline(logname, i)
    if len(stack) != 0:
        print(filename)

def getInput(data_str):
    data_str = re.sub('@@\\{.*?@@}', 'object', data_str)
    data_str = data_str.replace('@@{','')
    data_str = data_str.replace('@@}', '')
    data_str = data_str.replace('##[','[')
    data_str = data_str.replace('##]', ']')
    data_str = data_str.replace('#[','[')
    data_str = data_str.replace('#]', ']')
    data_str = data_str.replace('#;',';')
    data_str = data_str.replace('#,', ',')
    data_elems = data_str.split('@@;')
    values = '{'
    sep=''
    for elem in data_elems:
        if elem != '':
            value = elem.split('@@,')[1]
        else:
            break
        values += sep + value
        sep = ';'
    values += '}'
    return values

def getOutput(data_str):
    data_str = re.sub('@@\\{.*?@@}', 'object', data_str)
    data_str = re.sub('@@;','',data_str)
    return data_str.split('@@,')[1]

args = sys.argv
readlog(args[1])
# readlog('C:\H\Java_codes\output\solution12_small\solution12_small.txt')
# readlog('solution12_small.txt')