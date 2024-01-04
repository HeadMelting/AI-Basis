# 파이썬 함수

def change(a):
    a = 10
    print(id(a))
    

def sum(a,b):
	result = a + b
	return result

def noReturn(a,b):
    print(f"{a}와 {b}의 합은 {a+b}입니다")


res = sum(1,2)
print(res)

a = 23;
print(a)
change(a)
print(a, id(a))
noReturn(3,5)

def sum_may(*args):
	sum  = 0
	for i in args:
		sum = sum + i
	return sum

print(sum_may(1,24,125,1234,123,1234,123,1231,231,213,2,13,1,14,23,5))

def inputdic(**dic):
    print(dic)

inputdic(name="hyun",age="23")


# 튜플 형태로 리턴
def sum_and_mul(a,b):
    return a+b, a*b

def greetings(name,age,sayhi="hi"):
	print(sayhi,f", my name is {name}, and {age} years old")

greetings("hyunsu",23,"hoolw")
greetings(name="hyunsu",age=23,)

#lambda
def add(a,b):
    return a+b

add = lambda a, b : a+b

test = lambda *args : [i % 2 for i in args]

print(test(5,123,124,123,123,123,2,1,2,3,41,2,5,2))

funcList = [lambda a, b: a+b, lambda a,b : a*b]
funcList[0](1,2) #3
funcList[1](1,2) #2


hyunsu = {
    "sayhi" : lambda : print(f"{hyunsu['name']}, hi"),
    "name" : "hyunsui"
}

hyunsu["sayhi"]()
hyunsu["name"] = "moo"
hyunsu["sayhi"]()

a = input("이름을 입력하세요 : ")

hyunsu["name"] = a
hyunsu["sayhi"]()


## print
print("life" ,a, "is" "too short",end=" ")