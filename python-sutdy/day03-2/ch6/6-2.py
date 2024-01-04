#6-2
def Sum35(iter:int):
    result = 0;
    for i in range(1,iter):
        if i % 3 == 0 or i % 5 == 0 :
            result += i
    return result


print(Sum35(1000))
