from multiprocessing import Pool



def sumab(l):
    ret = l
    # for i in l:
    #     ret += i 
    return ret 
if  __name__ == "__main__":
    pool = Pool(processes=12)
    ret = pool.map(sumab, [1, 2])
    pool.close()
    pool.join()
    for m in ret:
        print(m)
# from multiprocessing import Pool

# def test(i):
#     print(i)
# if  __name__ == "__main__":
#     lists = [1, 2, 3]
#     pool = Pool(processes=2)     
#     pool.map(test, lists)          
#     pool.close()
#     pool.join()
