__author__ = 'LIMU_North'

import math

def count_inversion(lst):
    return merge_count_inversion(lst)[1]

def merge_count_inversion(lst):
    if len(lst) <= 1:
        return lst, 0
    middle = int( len(lst) / 2 )
    left, a = merge_count_inversion(lst[:middle])
    right, b = merge_count_inversion(lst[middle:])
    result, c = merge_count_split_inversion(left, right)
    return result, (a + b + c)

def merge_count_split_inversion(left, right):
    result = []
    count = 0
    i, j = 0, 0
    left_len = len(left)
    while i < left_len and j < len(right):
        if left[i] >= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            count += left_len - i
            j += 1
    result += left[i:]
    result += right[j:]
    return result, count

#### new standard: Kendall's tau

def Kendall_tau(lst):
    num_inversion = count_inversion(lst)
    if len(lst) <= 1:
        return 1
    else:
        #### Note: we are using the second definition of tau in Wiki.
        num_total_pairs = len(lst)*(len(lst)-1)/2.0

        itemlst = []

        ## eliminate pairs within same tier
        for item in list(set(lst)):
            if lst.count(item) == len(lst):
                print 'number of invertable pairs: ', 0
                print 'tau: ', 1
                return 1
            else:
                itemlst.append(lst.count(item))
                num_total_pairs -= lst.count(item)*(lst.count(item)-1)/2.0
        print lst
        print 'number of invertable pairs: ', num_total_pairs
        tmp_tau = (num_total_pairs-2*num_inversion)/float(num_total_pairs)
        print 'tau: ',tmp_tau
        return tmp_tau

if __name__ == "__main__":
    input_array_7 = [1,1,1,0]  #8
    print count_inversion(input_array_7)
    print Kendall_tau(input_array_7)

    print set(input_array_7)
    print len(list(set(input_array_7)))