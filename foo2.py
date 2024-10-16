dict = {"a": 1, "b": 2, "c": 3}

dict.keys()  # returns list of keys of dictionary
dict.values()  # returns list of values of dictionary
print(dict.get("z"))

a = [[9, 9, 8, 1], [5, 6, 2, 6], [8, 2, 6, 4], [6, 2, 2, 2]]

print(len(a) - 2, len(a[0]))


a[0][2 - 1 : 2 + 2].extend([1])

print(a)
print("--------")
s = ["a", "b"]
for i in range(-len(s), 0, 1):
    print(i)
