# Comprehension
list1 = [1, 2, 3, 4, 5]
list2 = [i for i in list1 if i % 2 == 0]
print(list2)

# Create a new list combining elements from list1 and list2
combined_list = [x + y for x in list1 for y in list2]
print("Combined list:", combined_list)

# Create a list of tuples pairing elements
paired_list = [(x, y) for x in list1 for y in list2]
print("Paired list:", paired_list)

# Create a dictionary mapping elements from list1 to list2
mapped_dict = {x: y for x in list1 for y in list2}
print("Mapped dictionary:", mapped_dict)

# Create a set of unique elements from list1
unique_set = set(list1)
print("Unique set:", unique_set)

