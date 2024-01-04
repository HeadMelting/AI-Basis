## function annotation
def complicated_func(param:list[str]) -> tuple[float,float,float]:
	return 2,4,1

# function annotation
def func(arg1:str,arg2:str,arg3:list[str or int]):
    '''
    param arg1:str
    :param arg2:str
    :param arg3: list
    :return: bool
    '''
    return True

# ==================================================================
## 클래스의 필요 이유
result1 = 0
result2 = 0

def add1(num:int or float) -> float:
    global result1
    result1 += num
    return result1

def add2(num):
    global result2
    result2 += num
    return result2

print(add1(3.2))
print(add1(4))

print(add2(3))
print(add2(7))

## 클래스

class Calculator:
    def __init__(self): # 생성자
        self.result = 0

    def add(self, num):
        self.result += num
        return self.result
    

cal1 = Calculator()
cal2 = Calculator()

print(cal1.add(3)) 
print(cal1.add(2))
print(cal2.add(4))
print(cal2.add(6))

## class2
class Fourcal:
    # 생성자
    def __init__(self,first,second): 
        #아무런 표시를 하지 않으면 public
        self.public = "public"
        #언더바 한개 : protected method, property
        self._protected = "protected"
        #언더바 두개 : private
        self.__private = "private"
        self.first = first
        self.second = second
        self.__result = self.__add()
    
    def add(self):
        result = self.first + self.second
        return result

    def __add(self):
        result = self.first + self.second  + 1
        return result
    
    def res(self):
        return self.__result
    

fc = Fourcal(1,4)
# print(fc.__add())
print(fc.add())
# print(fc.__result) # can't
print(fc.res())