# 6-1
def GuGU(num:int):
	# result = []
	# for i in range(1,10):
	# 	result.append(i*num)
    result = [num*i for i in range(1,10)]

    return result

print(GuGU(3))