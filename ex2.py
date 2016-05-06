import csv
import math
import random
import numpy as np
from numpy import distutils
# from sklearn.cross_validation import KFold
import networkx as nx
from random import shuffle


#cross validation function: reut
# 1. split the training set to train and validation sets.
# 2. run the model on the train set
# 3. validate the model on the validation set
def CrossValidation(Product_customer_rank_matrix, model_name ,k):
    indexes = [i for i in xrange(len(Product_customer_rank))]
    random.shuffle(indexes)

    slices = [indexes[i::k] for i in xrange(k)]
    if k == 1:
        flag = True

    sum_of_RMSE = 0
    for i in xrange(k):
        validation = slices[i]
        training = []
        for j in xrange(k):
            if j != i:
                training += slices[j]
        print('Start run the model: {}').format(model_name)
        Product_customer_rank_train, Product_customer_rank_test = \
            np.array([Product_customer_rank_matrix[j,:] for j in training]), \
            np.array([Product_customer_rank_matrix[l,:] for l in validation])
        Rank_train, Rank_test = Product_customer_rank_train[:,2], Product_customer_rank_test[:,2]
        estimated_ranks = model_name(Product_customer_rank_train, Rank_train, Product_customer_rank_test, flag)
        rTilda = np.subtract(Product_customer_rank_test[:,2], estimated_ranks[:,2])
        RMSE = evaluateModel(Product_customer_rank_test, estimated_ranks, validation)
        print('The RMSE of the model {} for iteration {} is: {}'). format(model_name, i, RMSE)
        Create_estimatedR_file(estimated_ranks, model_name, Product_customer_rank_test)
        sum_of_RMSE += RMSE

    print('The average RMSE of the model {} using cross-validation with {} folds is: {}').format(model_name, k, sum_of_RMSE/k)


# Create output file
def Create_estimatedR_file(estimated_ranks, model_name,Product_customer_rank_test,i):
    np.savetxt(model_name+"_"+i+"_estimatedRanks.csv", estimated_ranks, fmt='%s, %s, %s', delimiter=",",
               header='Product_ID,Customer_ID,Customer_estimated_rank', comments='')
    np.savetxt(model_name + "_" + i + "_realRanks.csv", Product_customer_rank_test, fmt='%s, %s, %s', delimiter=",",
               header='Product_ID,Customer_ID,Customer_real_rank', comments='')


def Create_regression_file(Rank_train, R_avg, Bu, Bi, )

# The base model which calculate r as: R_avg + Bu+ Bi
# Return a dictionary- for each (i,u) the value is the estimated rank
def base_model(Product_customer_rank_train, Rank_train, Product_customer_test, use_base_model = True, flag = False):
    R_avg_train = np.mean(Rank_train)
    Bu, Bi = minimizeRMSE_model(Product_customer_rank_train, Rank_train, R_avg_train)
    if use_base_model:
        estimated_ranks, estimated_parameters = estimatedRanks(Product_customer_test, R_avg_train, Rank_train, Bu, Bi)
        if flag:
            np.savetxt(model_name + "for_regression_check.csv", estimated_parameters, fmt='%s, %s, %s', delimiter=",",
                       header='product, user, rank, R_avg, Bu, Bi, neighbors_indications', comments='')
        return estimated_ranks
    else:
        return R_avg_train, Bu, Bi


# The model calculate r as: a*R_avg + b*Bu+ c*Bi + d*(1 if one of the neighbors has a rank with the user, 0 otherwise)
# Return a dictionary- for each (i,u) the value is the estimated rank
def graph_model(Product_customer_train, Rank_train, Product_customer_test, flag = False):
    R_avg_train, Bu, Bi = base_model(Product_customer_train, Rank_train, Product_customer_rank_test, False)
    Products_Graph = graph_creation()
    Neighbors_indications_dictionary = neighbors_indications(Products_Graph , Product_customer_train)
    estimated_ranks = estimatedRanks(Product_customer_test, R_avg_train, Bu, Bi,Neighbors_indications_dictionary, a, b, c, d)
    if flag:
        np.savetxt(model_name + "for_regression_check.csv", estimated_parameters, fmt='%s, %s, %s', delimiter=",",
                   header='product, user, rank, R_avg, Bu, Bi, neighbors_indications', comments='')
    return estimated_ranks


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


# Create an undirected graph of product1-product2
def graph_creation():
    with open('Network_arcs.csv', 'r') as csvfile:
        edges = list(csv.reader(csvfile))
        i = 0
        for products in edges:  # delete the header of the file
            if products[0] == 'Product1_ID':
                del edges[i]
                break
            i += 1
        Products_Graph = nx.DiGraph(edges)
    csvfile.close()
    return Products_Graph


#Create a dictionary: for each product and user = 1 if the user rank of the predecessors of the product, 0 otherwise
def neighbors_indications(Products_Graph, Product_customer_train, Product_customer_test):
    Neighbors_indications_dictionary = {}
    for product, user in Product_customer_test:
        for neighbor in Products_Graph.predecessors(str(product)):
            if (int(neighbor), user) in Product_customer_train:
                Neighbors_indications_dictionary[(product, user)] = 1
                break #one neighbor which the user has ranked it is enough- no need to check all the neighbors
    return Neighbors_indications_dictionary


# Calculate for each customer and for each product the Bu and Bi
def minimizeRMSE_model (Product_customer , Ranks, R_avg):
    return


# Calculate for each Product_customer couple the estimated value for the rank.
# Return a numpy array- [product, user, the estimated rank]
def estimatedRanks(Product_customer_test, R_avg, Bu, Bi, Neighbors_indications_dictionary, a=1, b=1, c=1, d=1):
    estimated_ranks = []
    estimated_parameters = []
    for product, user, rank in Product_customer_test:
        if (product, user) in Neighbors_indications_dictionary.keys():
            #Neighbors_indications_dictionary[(product, user)] = 1 if there is such a key
            estimated_ranks.append([product, user, a*R_avg + b*Bu[user] + c*Bi[product] + d])
            estimated_parameters.append([product, user, rank, R_avg, Bu[user], Bi[product], 1])
        else:
            estimated_ranks.append([product, user, a*R_avg + b*Bu[user] + c*Bi[product]])
            estimated_parameters.append([product, user, rank, R_avg, Bu[user], Bi[product], 0])
    return np.array(estimated_ranks), np.array(estimated_parameters)


# Evaluate the model: calculate the RMSE for the validation set
def evaluateModel(Product_customer_rank_test, estimated_ranks, validation):
    return np.sum(np.power(np.subtract(estimated_ranks[:, 2], Product_customer_rank_test[:, 2])), 2)


# returns two lists: of the customers and of the products which appear in the result file,
# -> meaning we need to estimate their ranking
# def RelCusPro():
#     relevant_products_customers = []
#     with open("results.csv","r") as csvfile:
#      reader = csv.DictReader(csvfile)
#      for row in reader:
#             relevant_products_customers.append([row['Product_ID'], row['Customer_ID'], 0])
#     csvfile.close()
#     # relevant_products_customers = list(set(relevant_products))
#     relevant_products = list(set(relevant_products))
#     relevant_customers = list(set(relevant_customers))
#     return relevant_products, relevant_customers


def main():
################### read P_C_matrix into numPy matrix ##################################
    with open('P_C_matrix.csv', 'r') as csvfile:
        input_matrix = list(csv.reader(csvfile))
        i = 0
        for products in matrix:  # delete the header of the file
            if products[0] == 'Product_ID':
                del matrix[i]
                break
            i += 1
        Product_customer_rank_matrix = np.array(input_matrix).astype('int')
    csvfile.close()
#######################################################################################

########################     Cross Validation Part    #################################
    for model_name in (graph_model, base_model):
        CrossValidation(Product_customer_rank_matrix, model_name, 3)
#######################################################################################

#############    check the coefficient using multiple regression   ####################
    CrossValidation(Product_customer_rank_matrix, graph_model, 1)
#######################################################################################

###################    call the results file as numpy array   #########################
    with open('results.csv', 'r') as csvfile:
        input_matrix = list(csv.reader(csvfile))
        i = 0
        for products in matrix:  # delete the header of the file
            if products[0] == 'Product_ID':
                del matrix[i]
                break
            i += 1
        Product_customer_results_matrix = np.array(input_matrix).astype('int')
    csvfile.close()
########################################################################################

#######  run the model #################################################################
    model_name = base_model #choose the model
    results_matrix = model_name(Product_customer_rank_matrix,
                                Product_customer_rank_matrix[:,2],
                                Product_customer_results_matrix)
#######################################################################################

#######  output file ###################################################################
    numpy.savetxt("EX2.csv", results_matrix, fmt='%s, %s, %s', delimiter=",",
                  header='Product_ID,Customer_ID,Customer_rank', comments='')
#######################################################################################

if __name__ == '__main__':
    main()
