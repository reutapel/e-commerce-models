import csv
import math
import random
import numpy as np
from numpy import distutils
# from sklearn.cross_validation import KFold
import networkx as nx
from random import shuffle
import time
import collections as coll
import statistics
import itertools
import logging


#cross validation function: reut
# 1. split the training set to train and validation sets.
# 2. run the model on the train set
# 3. validate the model on the validation set
def CrossValidation(Product_customer_rank_matrix, model_name ,k):
    print('{}: Start run cross validations on the model: {} with: {} folds'). \
        format((time.asctime(time.localtime(time.time()))), str(model_name)[10:21], k)
    logging.info('{}: Start run cross validations on the model: {} with: {} folds'. \
        format((time.asctime(time.localtime(time.time()))), str(model_name)[10:21], k))
    indexes = [i for i in xrange(len(Product_customer_rank_matrix))]
    random.shuffle(indexes)

    slices = [indexes[i::k] for i in xrange(k)]
    if k == 1:
        flag = True
    else:
        flag = False

    sum_of_RMSE = 0
    for i in xrange(k-1):
        validation = slices[i]
        if k == 1: #only 1 fold --> the training and the validation set are the same
            training = slices[i]
        else:
            training = []
            for j in xrange(k):
                if j != i:
                    training += slices[j]

        print('{}: Start run the model: {}, loop number {}').\
            format((time.asctime(time.localtime(time.time()))), model_name,i)
        logging.info('{}: Start run the model: {}, loop number {}'. \
            format((time.asctime(time.localtime(time.time()))), model_name, i))

        Product_customer_rank_train, Product_customer_rank_test = \
            np.array([Product_customer_rank_matrix[j,:] for j in training]), \
            np.array([Product_customer_rank_matrix[l,:] for l in validation])
        Rank_train, Rank_test = Product_customer_rank_train[:,2], Product_customer_rank_test[:,2]
        if i==(k-1):
            i = k-1
        estimated_ranks, estimated_ranks_product_user =\
            model_name(Product_customer_rank_train, Rank_train, Product_customer_rank_test, flag)
        estimated_ranks = calcFinalRank(estimated_ranks.T)
        final_estimated_ranks = estimated_ranks.astype(np.int)
        final_full_estimated_ranks = np.concatenate((estimated_ranks_product_user, final_estimated_ranks), axis=1)
        final_full_estimated_ranks.sort(axis = 0)
        Product_customer_rank_test.sort(axis = 0)
        rTilda = np.subtract(Product_customer_rank_test[:,2], final_full_estimated_ranks[:,2])
        RMSE = evaluateModel(Product_customer_rank_test, final_full_estimated_ranks)
        print('{}: The RMSE of the model {} for iteration {} is: {}').\
            format((time.asctime(time.localtime(time.time()))), model_name, i, RMSE)
        logging.info('{}: The RMSE of the model {} for iteration {} is: {}'. \
            format((time.asctime(time.localtime(time.time()))), model_name, i, RMSE))

        Create_estimatedR_file(final_full_estimated_ranks, model_name, Product_customer_rank_test,i)
        sum_of_RMSE += RMSE

    print('{}: The average RMSE of the model {} using cross-validation with {} folds is: {}').\
        format((time.asctime(time.localtime(time.time()))), model_name, k, sum_of_RMSE/k)
    logging.info('{}: The average RMSE of the model {} using cross-validation with {} folds is: {}'. \
        format((time.asctime(time.localtime(time.time()))), model_name, k, sum_of_RMSE / k))


#changes scale of ranks to 0-5, rounds and cast to int in estimated ranks
def calcFinalRank(estimated_ranks):
    OldMin = np.nanmin(estimated_ranks)
    OldMax = np.nanmax(estimated_ranks)
    OldRange = (OldMax - OldMin)
    if OldRange == 0:
        OldRange = 1
    i = 0
    for obs in estimated_ranks:
        OldRank = obs
        NewRank = (((OldRank - OldMin) * 5) / OldRange)
        Final = int(round(NewRank))
        estimated_ranks[i] = Final
        i += 1
    return estimated_ranks


# Create output file
def Create_estimatedR_file(full_estimated_ranks, model_name,Product_customer_rank_test,i):
    model_name= str(model_name)[10:21]
    np.savetxt(str(model_name) + "_" + str(i)+"_estimatedRanks.csv", full_estimated_ranks, fmt='%s, %s, %s', delimiter=",",
               header='Product_ID,Customer_ID,Customer_estimated_rank', comments='')
    np.savetxt(str(model_name) + "_" + str(i) + "_realRanks.csv", Product_customer_rank_test, fmt='%s, %s, %s', delimiter=",",
               header='Product_ID,Customer_ID,Customer_real_rank', comments='')


# The base model which calculate r as: R_avg + Bu+ Bi
# Return a dictionary- for each (i,u) the value is the estimated rank
def base_model(Product_customer_rank_train, Rank_train, Product_customer_rank_test, use_base_model = True, flag = False):
    print('{}: Calculate R average, Bu and Bi').format(time.asctime(time.localtime(time.time())))
    logging.info('{}: Calculate R average, Bu and Bi'.format(time.asctime(time.localtime(time.time()))))
    R_avg_train = np.mean(Rank_train)
    # rel_np_train_list, TestCustomers = relTrain(Product_customer_test, Product_customer_rank_train, R_avg_train)
    B_c, B_p = B_pc(Product_customer_rank_test, Product_customer_rank_train, R_avg_train)
    if use_base_model:
        estimated_ranks, estimated_ranks_product_user, estimated_parameters =\
            estimatedRanks(Product_customer_rank_test, R_avg_train, Rank_train, B_c, B_p, d=0)
        if flag:
            write_regression_parameters_to_file('base_model', estimated_parameters)
            # np.savetxt(model_name + "_for_regression_check.csv", estimated_parameters, fmt='%s, %s, %s', delimiter=",",
            #            header='product, user, rank, R_avg, Bu, Bi, neighbors_indications', comments='')
        return estimated_ranks, estimated_ranks_product_user
    else:
        return R_avg_train, B_c, B_p


# The model calculate r as: a*R_avg + b*Bu+ c*Bi + d*(1 if one of the neighbors has a rank with the user, 0 otherwise)
# Return a dictionary- for each (i,u) the value is the estimated rank
def graph_model(Product_customer_train, Rank_train, Product_customer_rank_test, flag = False):
    print('{}: Start run graph_model'.format(time.asctime(time.localtime(time.time()))))
    logging.info('{}: Start run graph_model'.format(time.asctime(time.localtime(time.time()))))
    R_avg_train, B_c, B_p = base_model(Product_customer_train, Rank_train, Product_customer_rank_test, False)
    Products_Graph_dic = graph_creation()
    Neighbors_average_rank_dictionary = neighbors_indications(Products_Graph_dic, Product_customer_train,
                                                             Product_customer_rank_test)
    estimated_ranks, estimated_ranks_product_user, estimated_parameters =\
        estimatedRanks(Product_customer_rank_test, R_avg_train, B_c, B_p, Neighbors_average_rank_dictionary)
    if flag:
        write_regression_parameters_to_file('graph_model', estimated_parameters)
        # np.savetxt(model_name + "_for_regression_check.csv", estimated_parameters, fmt='%s, %s, %s', delimiter=",",
        #            header='product, user, rank, R_avg, Bu, Bi, Neighbors_average_rank', comments='')
    return estimated_ranks, estimated_ranks_product_user

# Create an output file for the regression
def write_regression_parameters_to_file(model_name, estimated_parameters_list):
    with open(model_name + "_for_regression_check.csv", 'w') as write_file:
        writer = csv.writer(write_file, lineterminator='\n')
        fieldnames2 = ['product', 'user', 'rank', 'R_avg', 'Bu', 'Bi', 'Neighbors_average_rank']
        writer.writerow(fieldnames2)
        for estimated_parameters in estimated_parameters_list:
            writer.writerow([estimated_parameters[0], estimated_parameters[1], estimated_parameters[2],
                             estimated_parameters[3], estimated_parameters[4], estimated_parameters[5],
                             estimated_parameters[6]])
    write_file.close


def B_pc(np_test_list, np_train_list, rAvg):
    print('{}: Calculate Bu and Bi in B_pc function').format(time.asctime(time.localtime(time.time())))
    logging.info('{}: Calculate Bu and Bi in B_pc function'.format(time.asctime(time.localtime(time.time()))))
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
    print('{}: Finish calculate Bu and Bi in B_pc function').format(time.asctime(time.localtime(time.time())))
    logging.info('{}: Finish calculate Bu and Bi in B_pc function'.format(time.asctime(time.localtime(time.time()))))
    return B_c, B_p


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


# Create a dictionary of product1:product2 for the file Network_arcs
def graph_creation():
    print('{}: Start create the graph').format(time.asctime(time.localtime(time.time())))
    logging.info('{}: Start create the graph'.format(time.asctime(time.localtime(time.time()))))
    with open('Network_arcs.csv', 'r') as csvfile:
        edges = list(csv.reader(csvfile))
        i = 0
        for products in edges:  # delete the header of the file
            if products[0] == 'Product1_ID':
                del edges[i]
                break
            i += 1

        Products_Graph_dic = coll.defaultdict(list) #the dictionary is: product_2:product_1
        for edge in edges:
            Products_Graph_dic[int(edge[1])].append(int(edge[0]))
    return Products_Graph_dic


#Create a dictionary: for each product and user = 1 if the user rank of the predecessors of the product, 0 otherwise
def neighbors_indications(Products_Graph_dic, Product_customer_train, Product_customer_test):
    print('{}: Start find the neighbors indication').format(time.asctime(time.localtime(time.time())))
    logging.info('{}: Start find the neighbors indication'.format(time.asctime(time.localtime(time.time()))))
    Neighbors_average_rank_dictionary = {}
    all_product_rank = Product_customer_train[:, [0, 2]]
    sorted_products, Pidx, Pcnt = np.unique(all_product_rank[:, 0], return_inverse=True, return_counts=True)
    average_sorted_products = np.bincount(Pidx, weights=all_product_rank[:, 1]) / Pcnt
    product_average_dic = {}
    j = 0
    for product in sorted_products:
        product_average_dic[product] = average_sorted_products[j]
        j+= 1

    # i = 0
    for product_user in Product_customer_test:
        sum_of_average = 0
        neighbors = Products_Graph_dic.get(product_user[0])
        if neighbors == None:
            Neighbors_average_rank_dictionary[product_user[0]] = 0
            continue
        # print('{}: check product_user number {}: product {} has {} neighbors').\
        #     format(time.asctime(time.localtime(time.time())), i,product_user[0], len(neighbors))
        # i+=1
        for neighbor in neighbors:
            if product_average_dic.get(neighbor) == None:
                continue
            sum_of_average += product_average_dic[neighbor]
        average_rank = sum_of_average/float(len(neighbors))
        Neighbors_average_rank_dictionary[product_user[0]] = average_rank
    return Neighbors_average_rank_dictionary


# Calculate for each Product_customer couple the estimated value for the rank.
# Return a numpy array- [product, user, the estimated rank]
def estimatedRanks(Product_customer_test, R_avg, B_c, B_p, Neighbors_average_rank_dictionary={},
                   a=0.9976187, b=0.3601571, c=0.9441741, d=-0.0024044):
    print('{}: Start estimate the rank based on the model').format(time.asctime(time.localtime(time.time())))
    logging.info('{}: Start estimate the rank based on the model'.format(time.asctime(time.localtime(time.time()))))
    full_estimated_ranks = {}
    estimated_parameters = []

    #insert 0 to each product which is not in B_p
    test_products = Product_customer_test[:, 0]
    is_p_in_Bp = np.in1d(test_products, np.array(B_p.keys()))
    p_not_in_Bp = test_products[np.logical_not(is_p_in_Bp)]
    for product in p_not_in_Bp:
        B_p[product] = 0

  # insert 0 to each product which is not in B_c
    test_users = Product_customer_test[:, 1]
    is_c_in_Bc = np.in1d(test_users, np.array(B_c.keys()))
    c_not_in_Bc = test_users[np.logical_not(is_c_in_Bc)]
    for user in c_not_in_Bc:
        B_c[user] = 0

    print('{}: Start insert the estimated rank').format(time.asctime(time.localtime(time.time())))
    logging.info('{}: Start insert the estimated rank'.format(time.asctime(time.localtime(time.time()))))
    for product_user_rank in Product_customer_test:
        if product_user_rank[0] in Neighbors_average_rank_dictionary.keys():
            full_estimated_ranks[(product_user_rank[0], product_user_rank[1])]=\
                a*R_avg + b*B_p.get(product_user_rank[0]) + c*B_c.get(product_user_rank[1]) +\
                d*Neighbors_average_rank_dictionary[product_user_rank[0]]
            estimated_parameters.append([product_user_rank[0], product_user_rank[1], product_user_rank[2], R_avg,
                                         B_p.get(product_user_rank[0]), B_c.get(product_user_rank[1]),
                                         Neighbors_average_rank_dictionary[product_user_rank[0]]])
        else:
            full_estimated_ranks[(product, user)] = a*R_avg + b*B_p.get(product_user_rank[0]) + c*B_c.get(product_user_rank[1])
            estimated_parameters.append([product, user, product_user_rank[2], R_avg, B_p.get(product_user_rank[0]),
                                         B_c.get(product_user_rank[1]), 0])
    print('{}: Finish insert the estimated rank').format(time.asctime(time.localtime(time.time())))
    logging.info('{}: Finish insert the estimated rank'.format(time.asctime(time.localtime(time.time()))))
    estimated_ranks_product_user = full_estimated_ranks.keys()
    estimated_ranks = [full_estimated_ranks.values()]
    estimated_ranks_product_user = np.array(estimated_ranks_product_user, dtype=int)
    estimated_ranks = np.array(estimated_ranks)
    return estimated_ranks, estimated_ranks_product_user, estimated_parameters


# Evaluate the model: calculate the RMSE for the validation set
def evaluateModel(Product_customer_rank_test, estimated_ranks):
    print('{}: Start evaluate the RMSE').format(time.asctime(time.localtime(time.time())))
    logging.info('{}: Start evaluate the RMSE'.format(time.asctime(time.localtime(time.time()))))
    Product_customer_rank_test.sort(axis = 0)
    estimated_ranks.sort(axis = 0)
    np_size = Product_customer_rank_test.size/3.0
    C = np.full((np_size, 1), np_size)
    first_step = np.subtract(estimated_ranks[:,2], Product_customer_rank_test[:, 2])
    second_step = np.power(first_step,2)
    # second_step = second_step.T
    # i=0
    # for obs in second_step:
    #     second_step[i] = float(obs)/np_size
    #     i += 1
    # third_step = np.divide(second_step, C)
    third_step = np.sum(second_step)
    forth_step = third_step/np_size
    RMSE = np.sqrt(forth_step)
    # RMSE = np.sqrt(np.sum(np.divide(np.power(np.subtract(estimated_ranks[:,2], Product_customer_rank_test[:, 2]), 2), C)))
    return RMSE

def main():
    logging.basicConfig(filename='logfileRegression.log', level=logging.DEBUG)
################### read P_C_matrix into numPy matrix ##################################
    with open('P_C_matrix.csv', 'r') as csvfile:
        input_matrix = list(csv.reader(csvfile))
        i = 0
        for products in input_matrix:  # delete the header of the file
            if products[0] == 'Product_ID':
                del input_matrix[i]
                break
            i += 1
        Product_customer_rank_matrix = np.array(input_matrix).astype('int')
    csvfile.close()
#######################################################################################

########################     Cross Validation Part    #################################
    for model_name in (graph_model, base_model):
        CrossValidation(Product_customer_rank_matrix, model_name, 10)
#######################################################################################

#############    check the coefficient using multiple regression   ####################
    # CrossValidation(Product_customer_rank_matrix, graph_model, 1)
#######################################################################################

###################    call the results file as numpy array   #########################
    # with open('results.csv', 'r') as csvfile:
    #     input_matrix = list(csv.reader(csvfile))
    #     i = 0
    #     for products in matrix:  # delete the header of the file
    #         if products[0] == 'Product_ID':
    #             del matrix[i]
    #             break
    #         i += 1
    #     Product_customer_results_matrix = np.array(input_matrix).astype('int')
    # csvfile.close()
########################################################################################

#######  run the model #################################################################
    # model_name = base_model #choose the model
    # estimated_ranks,  = model_name(Product_customer_rank_matrix,
    #                             Product_customer_rank_matrix[:,2],
    #                             Product_customer_results_matrix)
    # estimated_ranks = calcFinalRank(estimated_ranks.T)
    # final_estimated_ranks = estimated_ranks.astype(np.int)
    # results_matrix = np.concatenate((estimated_ranks_product_user, final_estimated_ranks), axis=1)
    # results_matrix.sort(axis=0)
#######################################################################################

#######  output file ###################################################################
    # numpy.savetxt("EX2.csv", results_matrix, fmt='%s, %s, %s', delimiter=",",
    #               header='Product_ID,Customer_ID,Customer_rank', comments='')
#######################################################################################

if __name__ == '__main__':
    main()
