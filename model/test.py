import torch
import sys
def test():
    file_path = "C:\H\Java_codes\output\solution12_small\diagram.dot"
    res = []
    with open(file_path, 'r') as fin:
        lines = fin.read()
        line = lines.split("\"")
        cnt = 0
        for parts in line:
            # print(parts)
            if cnt % 2 == 1 and parts != ' ----> ':
                part = parts.split("\n")
                res.append(part[1])
            cnt += 1    

    print(res)
    return res
# print("1")
# print('num:' + str(len(sys.argv)))
# print(str(sys.argv))
ans = []
ans = test()