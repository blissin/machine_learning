import random

a=random.randint(1,100)
cnt=0
print("게임을 시작합니다. 1~100까지의 값을 입력해주세요.")
while True:
    cnt+=1
    b=int(input("입력: "))
    if a==b:
        print(f"정답입니다. {cnt}번만에 맞추셨습니다.")
        break
    elif a>b:
        print(f"{b}보다는 큽니다.")
        continue
    else:
        print(f"{b}보다는 작습니다.")
        continue