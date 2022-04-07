import json
import os
import subprocess

file_content = ''
with open('C:\H\Java codes\jsonsave\data1.json') as fin:
    file_content = fin.read()
tree = json.loads(file_content)

labels = []
edges = []
global_id = 0

def dfs(root):
    global global_id
    labels.append(root['label'])
    root_id = global_id
    global_id += 1
    for child in root['children']:
        child_id = global_id
        edges.append([root_id, child_id])
        dfs(child)

dfs(tree)
outputs = []
out_edges = []
for pairs in edges:
    if pairs[0] == 0:
        continue
    else:
        str1 = labels[pairs[0]]
        str2 = labels[pairs[1]]
        out_edges.append([pairs[0], pairs[1]])
        outputs.append([str1, str2])
node_map = []
num = 0
for pairs in out_edges:
    num = max(num, pairs[0], pairs[1])
for i in range(num):
    for j in range(len(outputs)):
        if out_edges[j][0] == i + 1:
            node_map.append(outputs[j][0])
            break
        if out_edges[j][1] == i + 1:
            node_map.append(outputs[j][1])
            break
def toDot():
    with open('C:\H\Java codes\jsonsave\diagram.dot', 'w') as f:
        f.write("digraph Process_Tree {\n")
        f.write("node [shape=box];\n")
        point_id = 0
        node_id = 0
        for node in node_map:
            if node != '':
                f.write(str(point_id) + " [label=\"node_id = " + str(node_id) + "\n" + node + "\"];\n")
                node_id += 1
            else:
                f.write(str(point_id) + " [label=\" ----> \"];\n")
            point_id += 1
        for pairs in out_edges:
            f.write(str(pairs[0] - 1) + " -> " + str(pairs[1] - 1) + ";\n")
        f.write("}")
toDot()
cmd2 = "cd C:\H\Java codes\jsonsave"
cmd3 = "dot -Tpng diagram.dot -o diagram.png"
cmd = cmd2 + '&&' + cmd3
subprocess.Popen(cmd, shell = True)
subprocess.call(cmd, shell = True)
# dot -Tpng diagram.dot -o diagram.png
# print(labels)
# print(edges)
# print(out_edges)
# print(outputs)
# print(node_map)