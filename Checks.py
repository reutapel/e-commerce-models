__author__ = 'shimon'
import csv
import math
import random
import time
import numpy as np


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

    test_list = [[12,34,0],[1,23,0],[1,34,0],[22,22,0]]
    train_list = [[12,12,6],[1,42,4],[1,7,5],[22,4,9],[22,43,6],[13,42,5]]
    np_test_list = np.array(test_list)
    print(np_test_list)
    np_train_list = np.array(train_list)
    print(np_train_list)

def minB(test_list, train_list):
    VectorBindex = {}
    TestProducts = np.array(test_list)[:,0]
    TestProducts = set(TestProducts)
    TestProductsLen = len(TestProducts)
    TestCustomers = np.array(train_list)[:,1]
    TestCustomers = set(TestCustomers)
    TestCustomersLen = len(TestCustomers)
    for obs in train_list:
        if (obs[0] in TestProducts) or (obs[1] in TestCustomers):
            continue
        else:











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
