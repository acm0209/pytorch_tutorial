import torch
import numpy as np

def line():
    for i in range(30):
        print("-", end="")
    print()
    print()

t = torch.FloatTensor([[1., 2., 3.],
                       [4., 5., 6.],
                       [7., 8., 9.],
                       [10., 11., 12.]
                      ])

print("텐서 슬라이싱")
line()

print(t[:, 1])
#1열 전체 출력
print(t[2:,1])
#2행부터 1열 전체 출력
print(t[:2,1])
#(2-1)행까지 1열 전체 출력
line()

print("텐서 브로드캐스팅")
line()

m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([[3], [4]])
print(m1.shape)
print(m2.shape)
print(m1 + m2)
# shape가 달라도, 차원을 자동으로 맞춰서 연산해 준다.
line()

print("평균 구하기")
line()

t = torch.FloatTensor([[1, 2], [3, 4]])
print(t.mean(dim=0))
# 차원별로 평균값을 구해냄.
line()

print("최대값 구하기")
line()

print(t.max(dim=0))
print('Max: ', t.max(dim=0)[0])
print('Argmax: ', t.max(dim=0)[1])
#최대값의 인덱스를 반환
line()

print("텐서 차원의 크기 변환")
line()
t = np.array([[[0, 1, 2],
               [3, 4, 5]],
              [[6, 7, 8],
               [9, 10, 11]]])
ft = torch.FloatTensor(t)
print(ft.view([-1, 3])) # ft라는 텐서를 (?, 3)의 크기로 변경
#3차원에서 2차원으로 변환
line()

print("차원 줄이기")
line()
print(ft.squeeze())
#인자를 주지 않으면 1인 모든 차원을 삭제
line()

print("차원 늘리기")
line()
print(ft.unsqueeze(0))
# 0번째 차원에 1차원 추가
line()

print("연결하기")
line()
x = torch.FloatTensor([[1, 2], [3, 4]])
y = torch.FloatTensor([[5, 6], [7, 8]])
print(torch.cat([x, y], dim=0))
#0번째 차원을 기준으로 연결
line()

print("stacking 사용")
line()
x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])
print(torch.stack([x, y, z]))
print(torch.stack([x, y, z], dim=1))
# unsqueeze를 사용하지 않고 바로 연결 가능
line()

print("같은 크기의 텐서 만들기")
line()
x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
print(torch.zeros_like(x))
line()