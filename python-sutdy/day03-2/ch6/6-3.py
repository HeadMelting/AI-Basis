# 6-3
def getTotalPageNum(postNum,postPerPage=10):
    (totalPage,lastPagePost) = divmod(postNum,postPerPage)
    if lastPagePost != 0:
        totalPage += 1
    return totalPage , lastPagePost