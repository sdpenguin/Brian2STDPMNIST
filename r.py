# [("a", 4), ("b", 3), ("c", 2), ("a", 1)]


input = "aaaabbbcca"


# print ([(i,kl) for (i,kl) in enumerate(input)])
lista = []
count = {}
prev= 0
for (i,kl) in enumerate(input): 
    
    print (i,kl, prev)
    if prev != kl and i>0:
        count[kl] = kl  
        lista.append((prev,0))
    prev = kl


print (lista)