# 파일 쓰기

f = open("newFile.txt",'w')
for i in range(1,11):
    data = f"{i}번째 줄입니다 \n"
    f.write(data)
f.close

# 파일 읽기

f = open("newFile.txt",'r',encoding='UTF-8')
while True:
    line = f.readline()
    if not line: break
    print(line,end=" ")
f.close()

f = open("newFile.txt",'r',encoding='UTF-8')
lines = f.readlines()
for line in lines:
    print(line)
f.close()

f = open("newFile.txt",'r',encoding='UTF-8')
data = f.read()
print(data)
f.close()

# 추가모드

f = open("newFile.txt",'a',encoding='UTF-8')
for i in range(11,20):
    data = f"{i}번째 줄입니다.\n"
    f.write(data)
f.close()

f = open("newFile.txt",'r',encoding='UTF-8')
data = f.read()
print(data)
f.close()

# with
with open("foo.txt",'w') as f:
    f.write("life is too short, you need python")
# auto f.close()

#immutable
a = 1
def vartest(a):
	a = a + 1
vartest(a)
print(a)
# 변하지 않음


# mutable
b = [1,2,3]
def vartest2(b):
	b.append(4)
vartest2(b)
print(b)
# 변함