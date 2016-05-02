import csv
import math
import random

Product_customer_rank ={}
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
    fieldnames = ["Customer_ID", "Proudct_ID","Customer_rank"]
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
def KfoldDevision(dict, threshold ):
    traindic = {}
    testdic = {}
    for key, value in dict:
        if random.random() <= threshold:
            traindic[(key[0], key[1])] = value
        else:
            testdic[(key[0], key[1])] = value
    return traindic, testdic
















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
with open('EX2.csv', 'w' ) as write_file:e
    writer = csv.writer(write_file, lineterminator='\n')
    fieldnames2 = ["Proudct_ID" , "Customer_ID" ,"Customer_rank"]
    writer.writerow(fieldnames2)
    for result in results_list:
        writer.writerow([  result[0] , result[1] , int(result[2])  ])
write_file.close
#######################################################################################
#######################################################################################

y=1
        