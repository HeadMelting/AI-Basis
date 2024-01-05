import numpy as np
import numpy.typing as npt
# TODO : Reduced Row echelon Form 
#      - Minus 1 trick

# 가우시안 소거법
# Row Echelon Form으로 변환
# Linear equation의 해를 구함.
def GaussianElimination(AMatrix:npt.NDArray, BMatrix:npt.NDArray):
    num_rows, num_cols = AMatrix.shape
    """
    Ax = B 형태의 linear equation의 solution을 return하는 함수.
    """ 
    

    def _sort(AMatrix,BMatrix):
        """
        AMatrix를 Pivot index 순서로 ascending sort
        """
        pivot_idx_arr = []
        for row in range(num_rows):
            idx, _ = _find_pivot(AMatrix[row])
            pivot_idx_arr.append(idx)
        pivot_idx_arr = np.array(pivot_idx_arr)
        sorted_indices = np.argsort(pivot_idx_arr)

        sorted_AMatrix = AMatrix[sorted_indices]
        sorted_BMatrix = BMatrix[sorted_indices]

        return sorted_AMatrix, sorted_BMatrix
    


    def _find_pivot(arr:npt.NDArray):
        """
        1d array를 input으로 받아서, 그 행의 피벗 값과 인덱스를 리턴.
        idx = -1 일 경우, 전부다 0인 행렬
        """
        idx = 9999999
        value = 0
        for col in range(num_cols):
            if arr[col] != 0:
                idx = col
                value = arr[col]
                break

        return idx, value


    def _make_row_echelon_form(AMatrix,BMatrix):
        """
        Reduced Row Echelon Form으로 변환
        """
        AMatrix, BMatrix = _sort(AMatrix,BMatrix)
        REF = np.zeros_like(AMatrix)
        REF_b = BMatrix
        print(AMatrix,BMatrix)
        for row in range(num_rows-1):
            # 맨 앞을 1로 만듬
            prev_p_idx, prev_p_value = _find_pivot(AMatrix[row])
            REF[row] = AMatrix[row]/prev_p_value
            REF_b[row] = BMatrix[row]/prev_p_value

            print('=== row#',row,"===================================================")
            for next_row in range(row+1,num_rows):
                p_idx, p_value = _find_pivot(AMatrix[next_row])
                if p_idx == prev_p_idx:
                    print('REF[',next_row,'] = AMatrix[',next_row,'] - ', p_value, '* REF[',row,']' ,'=', AMatrix[next_row], ' - ', p_value, '*', REF[row])
                    REF[next_row] = AMatrix[next_row] - p_value*REF[row]
                    REF_b[next_row] = BMatrix[next_row] - p_value*REF_b[row]
                else:
                    REF[next_row] = AMatrix[next_row]
                    REF_b[next_row] = BMatrix[next_row]

            REF, REF_b = _sort(REF, REF_b)
            AMatrix = REF
            BMatrix = REF_b
            print('REF:\n',REF,'\n================================================================\n')   
            print('REFb:\n',REF_b,'\n================================================================\n')   
        
        print(REF,REF_b) 
        return REF,REF_b

    def _particular_solution(AMat, BMat):
        x = np.zeros(AMat.shape[1])
        p_indics = []
        for row in range(num_rows):
            indic = num_rows - row - 1
            p_idx, _ = _find_pivot(AMat[indic])
            if p_idx != 9999999:
                p_indics.append(p_idx)
            else:
                p_indics.append(-1)
        
        print('p_indics',p_indics)

        all_columns = set(range(num_cols))
        for num in p_indics:
            all_columns.discard(num)
        
        free_variables = list(all_columns)


        for row in range(1, num_rows):
            if p_indics[row] == -1 :
                continue
            if row == 0 :
                factor = 0
            else:
                factor = np.dot(AMat[num_rows - 1 - row,p_indics[row]+1:], x[p_indics[row]+1:])
                print('factor = np.dot(',AMat[num_rows - 1 - row,p_indics[row]+1:],',',x[p_indics[row]+1:],')')
            x_next = BMat[num_rows - 1 - row] - factor
            print(x_next, '=', BMat[num_rows -1 - row], '-' ,factor)
            print(x_next)
            x[p_indics[row]] = x_next
        print(x)
        return x, free_variables
    
    def _main():
    #    print(AMatrix, BMatrix)
       REF,REF_b=_make_row_echelon_form(AMatrix,BMatrix)
       particular_sol, free_var = _particular_solution(REF,REF_b)
       print(particular_sol, free_var)

    _main()



# AMatrix = np.zeros([6,5])
BMatrix = np.array([-3,0,2,-1],dtype=np.float32)

# for i in range(6):
#     AMatrix[i,i:] += i

# AMatrix[0] = np.array([1,2,3,4,5])
# AMatrix = AMatrix[np.array([3,4,1,2,0,5])]

AMatrix = np.array(
[[-2, 4, -2, -1, 4,],
[1, -2, 1, -1, 1,],
 [4, -8, 3, -3, 1,],
 [1, -2, 0, -3, 4,]],dtype=np.float32)

GaussianElimination(AMatrix, BMatrix)