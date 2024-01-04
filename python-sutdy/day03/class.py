class Cal:
    def __init__(self,f,s):
        self.first = f
        self.second = s
    
    def add(self):
        result = self.first + self.second
        return result

class ChildCal(Cal):
    pass

class PowCal(Cal):
    def pow(self):
        result = self.first ** self.second
        return result

cc = ChildCal(2,5)
print(cc.add())

pc = PowCal(2,10)
print(pc.pow())

# 메서드 오버라이딩
class OverCal(Cal):
    def add(self):
        result = self.first + self.second + 1
        return result
    

oc = OverCal(4,4)
print(oc.add()) # 9

# 클래스 변수와 객체 변수

class Class2():
    클래스변수 = 23;
    name = "asdf"
    def __init__(self):
        self.name="Hyunsu"
        # pass

a = Class2()

print(a.클래스변수) #23
print(a.name) # Hyunsu