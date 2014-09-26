__author__ = 'LIMU_North'

def merge_sort(li, c):
    if len(li) < 2: return li
    m = len(li) / 2
    return merge(merge_sort(li[:m],c), merge_sort(li[m:],c),c)

def merge(l, r, c):
    result = []
    while l and r:
        print 'l: ', l
        print 'r: ', r
        if l[0] > r[0]:
            s = l
            # c[0] += l[0] - r[0]
        else:
            s = r
        print 's: ', s
        result.append(s.pop(0))
        if (s == r):
            print 'len(l): ', len(l)
            c[0] += len(l)
        # if (s == r): c += len(l)
    result.extend(l if l else r)
    return result

def easy_merge_sort(lst, count):
    count_list = [count]
    lst_output =  merge_sort(lst, count_list)
    return lst_output, count_list[0]

def BubbleSort2(lst):
    inverse_penalty = 0.0
    mylist = list(lst)
    swapped = True
    while swapped:
        swapped = False
        for i in range(len(mylist)-1):
            if mylist[i] < mylist[i+1]:
                inverse_penalty += mylist[i+1] - mylist[i]
                mylist[i], mylist[i+1] = mylist[i+1], mylist[i]
                swapped = True
    return inverse_penalty, mylist


if __name__ == '__main__':
    unsorted = [0.1, 1.0, 0.9, 0.3]
    unsorted02 = [1.0, 0.9, 0.1, 0.3]
    unsorted03 = [1.0, 0.3, 0.9, 0.1]
    count = 0
    new_sort, count =  easy_merge_sort(unsorted, count)
    # print merge_sort(unsorted, count)
    print count
    print new_sort

    print BubbleSort2(unsorted)[0]
    print BubbleSort2(unsorted03)[0]
    # aaasorted = [1,2,4,3]
    #
    # ggg = [unsorted.index(i) for i in new_sort]
    # print 'ggg: ', ggg
    # new_count = [0]
    # print merge_sort(ggg,new_count)
    # print 'sdfasdfasdf: ', new_count[0]
    # print easy_merge_sort(ggg,0)[1]
    #
    # print len(aaasorted[0:3])


