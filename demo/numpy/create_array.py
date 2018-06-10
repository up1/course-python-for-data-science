from timeit import default_timer as timer

start = timer()
my_list = list(range(1000000))
for _ in range(10): my_list2 = [x * 2 for x in my_list]
end = timer()
print(end - start)
