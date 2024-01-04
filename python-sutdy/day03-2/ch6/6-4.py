
# #6-4
# while True:
#     print("""
#     [1] 메모장 생성
#     [2] 메모장 읽기
#     [3] 메모장 추가
#     [4] 종료
#     """)

#     mode = input("원하는 모드에 맞는 숫자 입력: ")

#     if mode not in ["1","2","3","4"]:
#         print("없는 숫자입니다. 다시 입력해주세요")
#         continue
#     elif mode == "4":
#         print("종료합니다")
#         break
#     elif mode == "1":
#         while True:
#             title = input("메모장 파일 이름을 입력해주세요 : ")
#             try:
#                 f = open(f"{title}.txt",'w')
#                 break;
#             except:
#                 print("잘못된 파일 이름입니다.")
#                 continue
            
#         content = input("내용을 입력해주세요.")
#         f.write(content)
#         f.close()

#     elif mode == "2":
#         print("2")
#     elif mode == "3":
#         print("3")


import sys

option = sys.argv[1]
if option == "-a" :
    memo = sys.argv[2]
    f = open("memo.txt",'a')
    f.write(memo)
    f.write("\n")
    f.close()

elif option == "-v":
    f = open("memo.txt")
    memo = f.read()
    f.close()
    print(memo)