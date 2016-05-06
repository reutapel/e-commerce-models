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
    PCR_list = [[100,22,6],[6,42,4],[1,7,5],[22,4,9],[22,34,6],[13,62,5]]
    np_test_list = np.array(test_list)
    print(np_test_list)
    np_PCR_list = np.array(PCR_list)
    print(np_PCR_list)

#the following takes the relevant training observations that appears in test
def relTrain(np_test_list, np_train_list):
    TestProducts = np_test_list[:,0]
    TestProducts = list(set(TestProducts))
    TestProductsLen = len(TestProducts)
    TestCustomers = np_test_list[:,1]
    TestCustomers = list(set(TestCustomers))
    TestCustomersLen = len(TestCustomers)
    maskP = np.in1d(np_train_list[:, 0], TestProducts)
    maskC = np.in1d(np_train_list[:, 1], TestCustomers)
    mask = np.logical_or(maskP, maskC)
    rel_np_train_list = np_train_list[mask]
    return rel_np_train_list, TestProductsLen, TestCustomersLen

#returns two dictionaries of index in matrix for products and for customers, with the matrix itself initialized with zeros.
def buildIndexesForMatrix(np_PCR_list):
    Products = np_PCR_list[:, 0]
    Products = list(set(Products))
    ProductsLen = len(Products)
    Customers = np_PCR_list[:, 1]
    Customers = list(set(Customers))
    CustomersLen = len(Customers)
    CustomerMatrixIndex = {}
    ProductMatrixIndex = {}
    index = 0
    for product in Products:
        ProductMatrixIndex[product] = index
        index += 1
    index = 0
    for customer in Customers:
        CustomerMatrixIndex[customer] = index
        index += 1
    matrix = np.zeros(shape=(ProductsLen, CustomersLen))
    return ProductMatrixIndex, CustomerMatrixIndex, matrix




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
