#!/usr/bin/env python
# coding: utf-8

# In[120]:


import onnx


# In[121]:


import onnx
onnx_model = onnx.load("/workspace/android_sdk-rknntoolkit/rknn-toolkit2-1.3.0/bio_make_glass.onnx")
graph = onnx_model.graph


# In[122]:


nodes = graph.node
for i in range(len(nodes)):
    print(i,nodes[i])



###delete origin output and last node concat --huboheng
graph.output.remove(graph.output[0])
graph.node.remove(nodes[255])
#graph.node.remove(nodes[359])

# In[124]:

## 133 is number of class +5
out0 = onnx.helper.make_tensor_value_info('output0', onnx.TensorProto.FLOAT, [1,3,80,80,133])
out1 = onnx.helper.make_tensor_value_info('output1', onnx.TensorProto.FLOAT, [1,3,40,40,133])
out2 = onnx.helper.make_tensor_value_info('output2', onnx.TensorProto.FLOAT, [1,3,20,20,133])


# In[125]:


graph.output.extend([out0])
graph.output.extend([out1])
graph.output.extend([out2])


# In[126]:

'''
node = graph.node[336]
nodex = onnx.helper.make_node(
    'Transpose',
    inputs=['497'],
    outputs=['output2'],
    perm=[0,1,3,4,2]
)
graph.node.remove(node)
graph.node.insert(336, nodex)

node = graph.node[298]
nodex = onnx.helper.make_node(
    'Transpose',
    inputs=['451'],
    outputs=['output1'],
    perm=[0,1,3,4,2]
)
graph.node.remove(node)
graph.node.insert(298, nodex)

node = graph.node[260]
nodex = onnx.helper.make_node(
    'Transpose',
    inputs=['405'],
    outputs=['output0'],
    perm=[0,1,3,4,2]
)
graph.node.remove(node)
graph.node.insert(260, nodex)

for i in range(358,336,-1):
    try:
        graph.node.remove(nodes[i])
    except:
        print('no node:',i)

for i in range(320,298,-1):
    try:
        graph.node.remove(nodes[i])
    except:
        print('no node:',i)
for i in range(282,260,-1):
    try:
        graph.node.remove(nodes[i])
    except:
        print('no node:',i)
'''
# In[127]:
#'''
###delete every node from 3 transpose nodes to 3 last reshape nodes. Build new 3 transpose nodes which will give output.

## Transpose node 1
node = graph.node[238]
nodex = onnx.helper.make_node(
    'Transpose',
    inputs=['414'],
    outputs=['output2'],
    perm=[0,1,3,4,2]
)
graph.node.remove(node)
graph.node.insert(238, nodex)


## Transpose node 2
node = graph.node[219]
nodex = onnx.helper.make_node(
    'Transpose',
    inputs=['376'],
    outputs=['output1'],
    perm=[0,1,3,4,2]
)
graph.node.remove(node)
graph.node.insert(219, nodex)


## Transpose node 3
node = graph.node[200]
nodex = onnx.helper.make_node(
    'Transpose',
    inputs=['338'],
    outputs=['output0'],
    perm=[0,1,3,4,2]
)
graph.node.remove(node)
graph.node.insert(200, nodex)


#Removing nodes and outputs

for i in range(254,238,-1):
    try:
        graph.node.remove(nodes[i])
    except:
        print('no node:',i)

for i in range(235,219,-1):
    try:
        graph.node.remove(nodes[i])
    except:
        print('no node:',i)
for i in range(216,200,-1):
    try:
        graph.node.remove(nodes[i])
    except:
        print('no node:',i)
#'''

# In[128]:


onnx.checker.check_model(onnx_model)


# In[129]:


onnx.save(onnx_model, '/workspace/android_sdk-rknntoolkit/rknn-toolkit2-1.3.0/bio_make_glass_modi.onnx')


# In[ ]:




