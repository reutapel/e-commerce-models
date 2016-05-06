import csv
import math
import random
import numpy
from numpy import distutils
from sklearn.cross_validation import KFold
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

    for i in xrange(k):
        validation = slices[i]
        training = []
        for j in xrange(k):
            if j != i:
                training += slices[j]
        print('Start run the model')
        Product_customer_rank_train, Product_customer_rank_test = \
            numpy.array([Product_customer_rank_matrix[j,:] for j in training]), \
            numpy.array([Product_customer_rank_matrix[l,:] for l in validation])
        Rank_train, Rank_test = Product_customer_rank_train[:,2], Product_customer_rank_test[:,2]
        estimated_ranks = model_name(Product_customer_rank_train, Rank_train)
        RMSE = evaluateModel(Product_customer_rank_test, Product_customer_rank_test, estimated_ranks, validation)
        print('The RMSE of the model on this test set is: {}'). format(RMSE)

# The base model which calculate r as: R_avg + Bu+ Bi
# Return a dictionary- for each (i,u) the value is the estimated rank
def base_model(Product_customer_rank_train, Rank_train, Product_customer_rank_test, use_base_model = True):
    R_avg_train = numpy.mean(Rank_train)
    Bu, Bi = minimizeRMSE_model(Product_customer_rank_train, Rank_train, R_avg_train)
    if use_base_model:
        estimated_ranks = estimatedRanks(Product_customer_rank_test, R_avg_train, Bu, Bi)
        return estimated_ranks
    else:
        return R_avg_train, Bu, Bi


# The model calculate r as: a*R_avg + b*Bu+ c*Bi + d*(1 if one of the neighbors has a rank with the user, 0 otherwise)
# Return a dictionary- for each (i,u) the value is the estimated rank
def graph_model(Product_customer_train, Rank_train, Product_customer_rank_test):
    R_avg_train, Bu, Bi = base_model(Product_customer_train, Rank_train, Product_customer_rank_test, False)
    Products_Graph = graph_creation()
    Neighbors_indications_dictionary = neighpors_indications(Products_Graph , Product_customer_train)
    estimated_ranks = estimatedRanks(Product_customer_rank_test, R_avg_train, Bu, Bi,Neighbors_indications_dictionary, a, b, c, d)
    return estimated_ranks


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
def neighpors_indications(Products_Graph, Product_customer_train, Product_customer_test):
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
# Return a list- [product, user, the estimated rank]
def estimatedRanks(Product_customer_rank_test, R_avg, Bu, Bi, Neighbors_indications_dictionary, a=1, b=1, c=1, d=1):
    estimated_ranks = []
    for product, user in Product_customer_rank_test:
        if (product, user) in Neighbors_indications_dictionary.keys():
            #Neighbors_indications_dictionary[(product, user)] = 1 if there is such a key
            estimated_ranks.append([product, user, a*R_avg + b*Bu[user] + c*Bi[product] + d])
        else:
            estimated_ranks.append([product, user,a*R_avg + b*Bu[user] + c*Bi[product]])
    return estimated_ranks


# Evaluate the model: calculate the RMSE for the test set
def evaluateModel (Product_customer_rank_test, estimated_ranks, test):
    return sum(pow((estimated_ranks[i][2] - Product_customer_rank_test[i][2]), 2) for i in test)


# average of dictionary values- for calculating the average rank
def AvgDicValues(dic):
    # itervalues is more efficient than .values() because it does not create temporary list of values,
    # -> in python 3 it is not needed
    SumValues = sum(dic.itervalues())
    AvgValues = float(SumValues/(len(dic)))
    return AvgValues


# returns two lists: of the customers and of the products which appear in the result file,
# -> meaning we need to estimate their ranking
def RelCusPro():
    relevant_products_customers = []
    with open("results.csv","r") as csvfile:
     reader = csv.DictReader(csvfile)
     for row in reader:
            relevant_products_customers.append([row['Product_ID'], row['Customer_ID'], 0])
    csvfile.close()
    # relevant_products_customers = list(set(relevant_products))
    return relevant_products_customers


# random.random() returns float between 0.0 to 1.0
# returns test and training dictionaries with key: product, customer and value: rank
def KfoldDevision(dict, threshold):
    traindic = {}
    testdic = {}
    for key, value in dict:
        if random.random() <= threshold:
            traindic[(key[0], key[1])] = value
        else:
            testdic[(key[0], key[1])] = value
    return traindic, testdic


def main():
    # type: () -> object
    Product_customer_rank = {}
    #{(P_i,C_j):rank , (P_m,C_n):rank , .....}

    Customr_product_rank = {}
    #{(C_i,P_j):rank , (C_m,P_n):rank , .....}

    customer_product_list = []
    #[(C_i,P_j), (C_m,P_n), .....]

    product_neighbors = {}
    #{product1:[product1_neighbor1 , product1_neighbor2, ...] , product2:[product2_neighbor1 , product2_neighbor2, ...], ... }


########################################################################################
########################################################################################
################### read P_C_matrix into dictionaries ####################################
    with open("P_C_matrix.csv","r") as csvfile:
        reader = csv.DictReader(csvfile)
        # remember : field_names = ['Product_ID', 'Customer_ID','Customer_rank' ]
        for row in reader:
            Product_customer_rank[(row['Product_ID'],row['Customer_ID'])] = int(row['Customer_rank'])
            Customr_product_rank[(row['Customer_ID'],row['Product_ID'])] = int(row['Customer_rank'])
            customer_product_list.append((row['Customer_ID'],row['Product_ID']))
    csvfile.close()
#######################################################################################
#######################################################################################


########################################################################################
########################################################################################
################### read P_C_matrix into numPy matrix ####################################
    with open('P_C_matrix.csv', 'r') as csvfile:
        matrix = list(csv.reader(csvfile))
        i = 0
        for products in matrix:  # delete the header of the file
            if products[0] == 'Product_ID':
                del matrix[i]
                break
            i += 1
        Product_customer_rank_matrix = numpy.array(matrix).astype('int')
    csvfile.close()
#######################################################################################
#######################################################################################


########################################################################################
########################################################################################
####### another view of P_C_matrix #####################################################
    customer_product_list.sort()
    with open('C_P_matrix.csv', 'w' ) as csvfile:
        writer = csv.writer(csvfile, lineterminator='\n')
        fieldnames = ["Customer_ID", "Product_ID","Customer_rank"]
        writer.writerow(fieldnames)

        for C_P in customer_product_list:
            writer.writerow([  C_P[0] , C_P[1], Customr_product_rank[C_P] ])
    csvfile.close()
#######################################################################################
#######################################################################################


########################################################################################
################### read Network_arcs into dictionaries ################################
#################  Remember : Network_arcs is a directed graph #########################
    with open("Network_arcs.csv","r") as csvfile2:
        reader2 = csv.DictReader(csvfile2)
        # remember : field_names = ['Product1_ID', 'Product2_ID']
        for row in reader2:
            if row['Product1_ID'] in product_neighbors:
                product_neighbors[row['Product1_ID']].append(row['Product2_ID'])
            else:
                product_neighbors[row['Product1_ID']] = [row['Product2_ID']]
    csvfile2.close()
#######################################################################################
#######################################################################################


########################################################################################
########################################################################################
############################       results        ######################################
    product_last_rank = {}
    for P_C in Product_customer_rank:
        product_last_rank[P_C[0]] = int(Product_customer_rank[P_C])

    results_list = []
    # remember : field_names = ['Product_ID', 'Customer_ID','Customer_rank' ]
    with open("results.csv","r") as read_file:
        reader3 = csv.DictReader(read_file)
        for row in reader3:
            if row['Product_ID'] in product_last_rank:
                results_list.append( (row['Product_ID'] ,row['Customer_ID'] , product_last_rank[( row['Product_ID'] )]  )  )
            else:
                results_list.append( (row['Product_ID'] ,row['Customer_ID'] , 5  )  )
    read_file.close()

########################################################################################
########################################################################################
#######  output file ###################################################################
    with open('EX2.csv', 'w' ) as write_file:
        writer = csv.writer(write_file, lineterminator='\n')
        fieldnames2 = ["Product_ID" , "Customer_ID" ,"Customer_rank"]
        writer.writerow(fieldnames2)
        for result in results_list:
            writer.writerow([  result[0] , result[1] , int(result[2])  ])
        write_file.close
#######################################################################################
#######################################################################################

if __name__ == '__main__':
    main()
