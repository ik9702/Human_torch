import numpy as np
import torch
from torch import nn




device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
# torch.nn.Conv2d(
#     in_channels,            #입력 데이터의 채널 수    
#     out_channels,           #출력 데이터의 채널 수
#     kernel_size,            #filter 크기
#     stride=1,               #합성곱시 필터가 움직이는 간격
#     padding=0,              #input을 둘러싸는 padding의 수
#     dilation=1,             #필터 사이 간격을 의미
#     groups=1,               #input channel과 output channel 대응 설정
#     bias=True,              #output channel의 가중치 적용 여부
#     padding_mode='zeros'    #input을 둘러싸는 값 설정
# )



# m1 = nn.Conv2d(16, 33, 3, stride=2)
# #3*3 필터, stride=(2,2)

# # m2 = nn.Conv2d(16,33,(3,5),stride=2,padding=(4,2))
# # #3*5 필터, stride=2, padding(4,2)

m3 = nn.Conv2d(3,10,3).to(device)
#3*5 필터, stride (2,1), paddint(4,2), dilation (3,1)

input = torch.randn(1,3,5,5).to(device)

output = m3(input)

print(output.shape)
