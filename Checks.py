__author__ = 'shimon'
import csv
import math
import random
import time
import numpy as np
import collections as coll


def main():
    # # type: () -> object
    # Product_customer_rank = {}
    # #{(P_i,C_j):rank , (P_m,C_n):rank , .....}
    #
    # Customr_product_rank = {}
    # #{(C_i,P_j):rank , (C_m,P_n):rank , .....}
    #
    # customer_product_list = []
    # #[(C_i,P_j), (C_m,P_n), .....]
    #
    # product_neighbors = {}
    # #{product1:[product1_neighbor1 , product1_neighbor2, ...] , product2:[product2_neighbor1 , product2_neighbor2, ...], ... }
    rAvg = 3
    test_list = [[12,34,0],[1,23,0],[1,34,0],[22,22,0]]
    train_list = [[100,22,2],[6,42,4],[1,7,5],[22,4,9],[22,34,6],[13,62,5],[190,62,3],[540,62,3],[2, 2, 4], [2, 14, 1]]
    np_train_list = np.array(train_list)
    np_test_list = np.array(test_list)
    # rel_product_rank = np_test_list[:, [0, 2]]
    # print(rel_product_rank)
    # customer_rank = np_test_list[:, [1, 2]]
    # # print(customer_rank)
    # unq, idx, cnt = np.unique(rel_product_rank[:, 0], return_inverse=True, return_counts=True)
    # avg = np.bincount(idx, weights=rel_product_rank[:, 1]) / cnt
    # print(avg)
    # print(unq)


    TestProducts = np_test_list[:,0]
    TestProducts = list(set(TestProducts))
    TestCustomers = np_test_list[:,1]
    TestCustomers = list(set(TestCustomers))
    maskP = np.in1d(np_train_list[:, 0], TestProducts)
    maskC = np.in1d(np_train_list[:, 1], TestCustomers)
    mask = np.logical_or(maskP, maskC)
    rel_np_train_list = np_train_list[mask]
    rel_product_rank = rel_np_train_list[:, [0, 2]]
    sorted_products, Pidx, Pcnt = np.unique(rel_product_rank[:, 0], return_inverse=True, return_counts=True)
    average_sorted_products = np.bincount(Pidx, weights=rel_product_rank[:, 1]) / Pcnt
    rel_customer_rank = rel_np_train_list[:, [1, 2]]
    sorted_customers, Cidx, Ccnt = np.unique(rel_customer_rank[:, 0], return_inverse=True, return_counts=True)
    average_sorted_customers = np.bincount(Cidx, weights=rel_customer_rank[:, 1]) / Ccnt
    B_p = {}
    B_c = {}
    i = 0
    for product in sorted_products:
        B_p[product] = average_sorted_products[i] - rAvg
        i += 1
    i = 0
    for customer in sorted_customers:
        B_c[customer] = average_sorted_customers[i] - rAvg
        i += 1
    print(B_p)
    print(B_c)



    # print(np_test_list)
    # np_PCR_list = np.array(PCR_list)
    # print(np_PCR_list)
    # rAvg = 3
    # B_Customers = coll.defaultdict(lambda: [0] * 2)
    # B_Products = coll.defaultdict(lambda: [0] * 2)
    # for obs in rel_np_train_list:
    #     B_Products[obs[0]][0] += 1
    #     B_Products[obs[0]][1] += obs[2]
    #     B_Customers[obs[1]][0] += 1
    #     B_Customers[obs[1]][1] += obs[2]
    # print(B_Customers)
    # for key in B_Products.keys():
    #     B_Products[key][1] = B_Products[key][1]/B_Products[key][0]-rAvg
    # for key in B_Customers.keys():
    #     B_Customers[key][1] = B_Customers[key][1]/B_Customers[key][0]-rAvg
    # print(B_Customers)



# the following takes the relevant training observations that appears in test
# def relTrain(np_test_list, np_train_list):
#     TestProducts = np_test_list[:,0]
#     TestProducts = list(set(TestProducts))
#     TestProductsLen = len(TestProducts)
#     TestCustomers = np_test_list[:,1]
#     TestCustomers = list(set(TestCustomers))
#     TestCustomersLen = len(TestCustomers)
#     maskP = np.in1d(np_train_list[:, 0], TestProducts)
#     maskC = np.in1d(np_train_list[:, 1], TestCustomers)
#     mask = np.logical_or(maskP, maskC)
#     rel_np_train_list = np_train_list[mask]
#     return rel_np_train_list, TestProductsLen, TestCustomersLen








#returns two dictionaries of index in matrix for products and for customers, with the matrix itself initialized with zeros.
# def buildIndexesForMatrix(np_PCR_list):
#     Products = np_PCR_list[:, 0]
#     Products = list(set(Products))
#     ProductsLen = len(Products)
#     Customers = np_PCR_list[:, 1]
#     Customers = list(set(Customers))
#     CustomersLen = len(Customers)
#     CustomerMatrixIndex = {}
#     ProductMatrixIndex = {}
#     index = 0
#     for product in Products:
#         ProductMatrixIndex[product] = index
#         index += 1
#     index = 0
#     for customer in Customers:
#         CustomerMatrixIndex[customer] = index
#         index += 1
#     matrix = np.zeros(shape=(ProductsLen, CustomersLen))
#     return ProductMatrixIndex, CustomerMatrixIndex, matrix
#



    # VectorBindex = {}
    # c_index = 0
    # p_index = 500000
    # with open("P_C_matrix.csv","r") as csvfile:
    #     reader = csv.DictReader(csvfile)
    #     # remember : field_names = ['Product_ID', 'Customer_ID','Customer_rank' ]
    #     time1 = time.time()
    #     for row in reader:
    #         Product_customer_rank[(row['Product_ID'],row['Customer_ID'])] = int(row['Customer_rank'])
    #         if row['Product_ID'] in VectorBindex.keys():
    #             continue
    #         else:
    #             VectorBindex[row['Product_ID']] = p_index
    #             p_index += 1
    #         if row['Customer_ID'] in VectorBindex.keys():
    #             continue
    #         else:
    #             VectorBindex[row['Customer_ID']] = c_index
    #             c_index += 1
    #     time2 = time.time()
    #     duration = time2 - time1
    #     print(duration)
    # csvfile.close()




    # allData_customers = []
    # allData_products = []
    # with open("P_C_matrix.csv","r") as csvfile:
    #  reader = csv.DictReader(csvfile)
    #  time1 = time.time()
    #  for row in reader:
    #         allData_products.append(row['Product_ID'])
    #         allData_customers.append(row['Customer_ID'])
    # csvfile.close()
    # allData_products = list(set(allData_products))
    # allData_customers = list(set(allData_customers))
    # c_index = len(allData_products)
    # p_index = len(allData_customers)
    # time2 = time.time()
    # duration = time2 - time1
    # print(duration)
    # return allData_products, allData_customers
# def RelCusPro():
#     relevant_products_customers = []
#     with open("results.csv","r") as csvfile:
#         reader = csv.DictReader(csvfile)
#         for row in reader:
#             relevant_products_customers.append([row['Product_ID'], row['Customer_ID'], 0])
#         csvfile.close()
#         # relevant_products_customers = list(set(relevant_products))
#         return relevant_products_customers


if __name__ == '__main__':
    main()
