import csv
import math
import random
import numpy
from numpy import distutils
from sklearn.cross_validation import KFold
import networkx as nx

#cross validation function: reut
# 1. split the training set to train and validation sets.
# 2. run the model on the train set
# 3. validate the model on the validation set
def CrossValidation (Product_customer_rank, model_name):
    kf = KFold(len(Customr_product_rank), 3, shuffle=True)
    for train, test in kf:
        print('Start run the model')
        Product_customer_train, Product_customer_test = [Product_customer_rank.keys()[i] for i in train], [Product_customer_rank.keys()[i] for i in test]
        Rank_train, Rank_test = [Customr_product_rank.itervalues()[i] for i in train], [Customr_product_rank.itervalues()[i] for i in test]
        estimated_ranks = model_name(Product_customer_train, Rank_train)
        RMSE = evaluatModel(Product_customer_test, Rank_test,estimated_ranks, test)
        print('The RMSE of the model on this test set is: {}'). format(RMSE)


# The base model which calculate r as: R_avg + Bu+ Bi
# Return a dictionary- for each (i,u) the value is the estimated rank
def base_model(Product_customer_train, Rank_train, Product_customer_test):
    R_avg_train = AvgDicValues(Rank_train)
    Bu, Bi = minimizeRMSE_model(Product_customer_train, Rank_train, R_avg_train)
    estimated_ranks = estimatedRanks(Product_customer_test, R_avg_train, Bu, Bi)
    return estimated_ranks


# The model calculate r as: a*R_avg + b*Bu+ c*Bi + d*(1 if one of the neighbors has a rank with the user, 0 otherwise)
# Return a dictionary- for each (i,u) the value is the estimated rank
def graph_model(Product_customer_train, Rank_train, Product_customer_test):
    R_avg_train = AvgDicValues(Rank_train)
    Bu, Bi = minimizeRMSE_model(Product_customer_train, Rank_train, R_avg_train)
    Products_Graph = graph_creation()
    Neighbors_indications_dictionary = neighpors_indications(Products_Graph , Product_customer_train)
    estimated_ranks = estimatedRanks(Product_customer_test, R_avg_train, Bu, Bi,Neighbors_indications_dictionary, a, b, c, d)
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
def neighpors_indications(Products_Graph, Product_customer):
    Neighbors_indications_dictionary = {}
    for product, user in Product_customer:
        for neighbor in Products_Graph.predecessors(str(product)):
            if (int(neighbor), user) in Product_customer:
                Neighbors_indications_dictionary[(product, user)] = 1
                break #one neighbor which the user has ranked it is enough- no need to check all the neighbors
    return  Neighbors_indications_dictionary


# Calculate for each customer and for each product the Bu and Bi
def minimizeRMSE_model (Product_customer , Ranks, R_avg):
    return


# Calculate for each Product_customer couple the estimated value for the rank.
# Return a list- for each (i,u) the value is the estimated rank
def estimatedRanks (Product_customer_test, R_avg, Bu, Bi, Neighbors_indications_dictionary, a=1, b=1, c=1, d=1):
    estimated_ranks = []
    for i,u in Product_customer_test.keys():
        if (i,u) in Neighbors_indications_dictionary.keys():
            estimated_ranks.append(a*R_avg + b*Bu[u] + c*Bi[i] + d*Neighbors_indications_dictionary[(i,u)])
        else:
            estimated_ranks.append(a * R_avg + b * Bu[u] + c * Bi[i])
    return estimated_ranks


# Evaluate the model: calculate the RMSE for the test set
def evaluateModel (Ranks_test, estimated_ranks, test):
    return sum(pow((estimated_ranks[i] - Ranks_test[i]), 2) for i in test)


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
    relevant_customers = []
    relevant_products = []
    with open("results.csv","r") as csvfile:
     reader = csv.DictReader(csvfile)
     for row in reader:
            relevant_products.append(row['Product_ID'])
            relevant_customers.append(row['Customer_ID'])
    csvfile.close()
    relevant_products = list(set(relevant_products))
    relevant_customers = list(set(relevant_customers))
    return relevant_products, relevant_customers


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
        fieldnames2 = ["Proudct_ID" , "Customer_ID" ,"Customer_rank"]
        writer.writerow(fieldnames2)
        for result in results_list:
            writer.writerow([  result[0] , result[1] , int(result[2])  ])
        write_file.close
#######################################################################################
#######################################################################################

if __name__ == '__main__':
    main()
