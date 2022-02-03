import torch
import numpy as np













# torch.nn.Conv2d(
#     in_channels, 
#     out_channels, 
#     kernel_size, 
#     stride=1, 
#     padding=0, 
#     dilation=1,     #atrous 알고리즘, 필터 사이 간격을 의미
#     groups=1,       #1:일반적인 합성곱, 2:2그룹으로 나누어 연산 후 결합(연쇄), 3ㅑㅜ_초무ㅜ딘:각 ㅇ니풋 채널이 아웃풋 채널과 대응되어 연산(size:out_channels//in_channel)
#     bias=True, 
#     padding_mode='zeros'
# )



m1 = torch.nn.Conv2d(16, 1, 3, stride=2)

m2 = torch.nn.Conv2d(16,1,(3,5),stride=2,padding=(4,2))

m3 = torch.nn.Conv2d(16,1,(3,5),stride=(2,1),padding=(4,2), dilation=(3,1))


input = torch.randn(20,16,50,100)

output = m1(input)

print(input, m1, m2, m3, output)



