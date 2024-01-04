#vars
a = 1
print(type(a)) #int, float

b = 4.24e10

#사칙연산
# +, - , /, * ,
# // 몫, ** 제곱, % 나머지

# 문자열
strd = ''' asdfasdf
asdf
asdf
asdf
as
f'''

print(strd[23])

strformat = 'acc = {acc:.4f}'.format(acc = 0.12319)

print(strformat)
print(strformat.find('=').__bool__())

### list
li = [1,3,4,5,7,8]

print(li.index(3))

# dictionary

dic = {
    "a":"A",
    1 : 1,
    "b" : 2,
    3 : "C",

}

print(dic.get(2,None))

# set

s1 = {1,2,4,3,4,5}
s1 = set(s1)

print(type(s1))

if 0:
    print("hi")

[a,b] = ["asdf","dfdf"]
print(type(a),b)

print(1 in li)

# bool
True and not False or True  and False

# 3항 연산자
# message = 'success' if a > 60 else "fail"

treeHit = 0

while treeHit < 10:
    treeHit = treeHit + 1

    if treeHit % 2 == 0 :
        continue
    
    print(f"나무를 총 {treeHit}번 쳤습니다")

    if treeHit == 10:
        print("나무가 쓰러졌습니다")


# for문

test_list = ['one','two','three']
test_tuple = ('one1','two1','three1')

for i in test_tuple:
    print(i)

tuple_in_list = [(1,2),(3,4),(5,6)]
for (first, last) in tuple_in_list:
    print(f"( {first}, {last} )")


marks = [90, 25, 67, 45, 80]
num_iter = 0;

for mark in marks :
    num_iter = num_iter + 1
    if mark >= 60:
        print(f"{num_iter}번 학생은 불합격입니다")
    else:
        print(f"{num_iter}번 학생은 합격입니다")


marks = [90, 25, 67, 45, 80]
num_iter = 0;

for (index, mark) in enumerate(marks) :
    num_iter = num_iter + 1
    if mark >= 60:continue
    print(f"{index}번 학생은 합격입니다")

sumation = 0
for i in range(1,10):
    sumation = sumation + i
    print(sumation)


result = [num * 3 for num in marks]
print(result,end=' ',)

x_mul_y = [x * y for x in range(2,10) for y in range(1,10)]
print(x_mul_y)