#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

# import re

# name = 'Jeffrey Skilling'
# regex = re.compile(' '.join(name.upper().split()[::-1]) + r'\s?\w?')

# for name, feature in enron_data.items():
#     matchobj = regex.search(name)
#     if matchobj:
#         print(name, feature['exercised_stock_options'])

# # Section 25
# ceo = 'Jeffrey Skilling'
# chairman = 'Kenneth Lay'
# cfo = 'Andrew Fastow'

# regex_ceo = re.compile(' '.join(ceo.upper().split()[::-1]) + r'\s?\w?')
# regex_chairman = re.compile(' '.join(chairman.upper().split()[::-1]) + r'\s?\w?')
# regex_cfo = re.compile(' '.join(cfo.upper().split()[::-1]) + r'\s?\w?')

# total_payments = 0
# for name, feature in enron_data.items():
#     match_ceo = regex_ceo.search(name)
#     match_chairman = regex_chairman.search(name)
#     match_cfo = regex_cfo.search(name)
    
#     if match_ceo or match_chairman or match_cfo:
#         print(name, feature['total_payments'])
#         total_payments += int(feature['total_payments'])

# print(total_payments)

# # Section 27
# salary_count = 0
# email_count = 0
# for name, feature in enron_data.items():
#     if feature['salary'] != 'NaN':
#         salary_count += 1
#     if feature['email_address'] != 'NaN':
#         email_count += 1

# print(salary_count)
# print(email_count)

# # Section 29
# nan_in_salary = 0
# for name, feature in enron_data.items():
#     if feature['total_payments'] == 'NaN':
#         nan_in_salary += 1

# print(nan_in_salary/len(enron_data)*100)

# # Section 31
# poi_count = 0
# nan_in_total_payments = 0
# for name, feature in enron_data.items():
#     if feature['poi']:
#         poi_count += 1
#         if feature['total_payments'] == 'NaN':
#             nan_in_total_payments += 1

# print((nan_in_total_payments)/(poi_count)*100)

# # Section 32
# nan_in_total_payments = 0
# for name, feature in enron_data.items():
#     if feature['total_payments'] == 'NaN':
#         nan_in_total_payments += 1

# print(len(enron_data) + 10)
# print(nan_in_total_payments + 10)

# Section 33
poi_count = 0
nan_in_total_payments = 0
for name, feature in enron_data.items():
    if feature['poi']:
        poi_count += 1
        if feature['total_payments'] == 'NaN':
            nan_in_total_payments += 1

print(poi_count + 10)
print(nan_in_total_payments + 10)
