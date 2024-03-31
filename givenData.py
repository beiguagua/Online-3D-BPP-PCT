# container_size: A vector of length 3 describing the size of the container in the x, y, z dimension.
# item_size_set:  A list records the size of each item. The size of each item is also described by a vector of length 3.

# container_size = [10,10,10]
container_size = [600,600,400]
# container_size = [1,1,1]

lower = 1
higher = 5
resolution = 1
item_size_set = []
# for i in range(lower, higher + 1):
#     for j in range(lower, higher + 1):
#         for k in range(lower, higher + 1):
#                 item_size_set.append((i * resolution, j * resolution , k *  resolution))

item_size_set.append((205,200,87))
item_size_set.append((245,175,133))
item_size_set.append((220,140,113))
item_size_set.append((155,155,53))
item_size_set.append((235,160,120))
item_size_set.append((200,130,97))

# If you want to sample item sizes from a uniform distribution in continuous domain,
# type --sample-from-distribution in your command line.