# pretrained-torch
#example 以resnet 为例 
# 加载预训练的model，只加载部分参数！！
model  = resnet() # 自己构建的model

model_dict = model.state_dict()# 你自己定义的model 即你自己想加载进去的参数 和权重值

pretraind_dict = torch.load('resnet.pkl')

pretraind_dict = {k:v for k,v in pretraind_dict.items() if k in model_dict}

model_dict.update(pretraind_dict)

model.laod_state_dict(model)
