# try:
#     4/0
# except ZeroDivisionError as e:
#     print(e)

# print("hi")

# try:
#     f = open("no_such_file.txt",'r')
# except FileNotFoundError as e:
#     print(str(e))
# else:
#     data = f.read()
#     print(data)
# finally:
#     f.close()


#raise

class Bird:
    def fly(self):
        raise NotImplementedError

class Eagle(Bird):
    def fly(self):
        print("shssos")

eval('1+2')

def isPositive(x):
	return x > 0

a = list(filter(lambda x : x > 0,[1,-3,2,0]))
print(a)