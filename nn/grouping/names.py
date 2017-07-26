#!/usr/bin/env python

# import numpy as np
# import pandas as pd

Names = [
    ["Marius","B"],
    ["Jonatan","C"],
    ["Robin Benjamin","F"],
    ["Jascha","G"],
    ["Christian","J"],
    ["Lars","L"],
    ["Timo","N"],
    ["Nadine","N"],
    ["Julian","O"],
    ["Ruben Maximilian","S"],
    ["Ruben Rolf","S"],
    ["Yanick Julian","S"],
    ["Henning Alexander","U"]
    ]

# Names2 = pd.DataFrame(Names3)

# def getName(index):
#     return names[index]



# f = lambda x: names[x]
# g = np.vectorize(f)
#
# def bob(A):
#     for line in A:
#         for element in line:
#             element = g(element)
# #            lambda element: IndexToName(element)
#
# a = [[1,2,3],[1,2]]
# a
# print(bob(a))
#
# import pandas as pd
#
# b = pd.DataFrame(a, dtype=np.int64)
# b
# b.info()
# b.count()
# type(b[0][0])
#
# for index, row in b.iterrows():
#     for e in row:
#         if e != np.nan:
#             try:
#                 print(type(e))
#                 print(names[e])
#             except:
#                 pass
#
# # b.replace(1, "bob1", inplace=True)
# b
# b[0][0]
#
# for col in b:
#     for element in b[col]:
#         if element != "NaN":
#             print(type(element))
#             element = names[element]
