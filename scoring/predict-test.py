#!/usr/bin/env python
# coding: utf-8

import requests
import json

with open('example_bad_customer.json', 'r') as file:
    good_customer = json.load(file)

with open('example_good_customer.json', 'r') as file:
    bad_customer = json.load(file)

url = 'http://localhost:9696/predict'
customer_id = 'test_customer'

examples = [good_customer, bad_customer]

for example in examples:
    response = requests.post(url, json=example).json()
    if response['customer_default'] == True:
        print("\n", "Approved customer - ID: %s" % example["customer_ID"])
        print(response)
    else:
        print("\n", "Not approved customer - ID: %s" % example["customer_ID"])
        print(response)
