import csv

data = [] 
hypo = []
with open("1.csv") as csv_file: 
   read = csv.reader(csv_file,delimiter = ',')
   print("The Given Traning examples Are given as follows ")

   for row in read:
    print(row)

    if row[len(row)-1].upper() == "YES":
        data.append(row)


hypo = data[0]   
k=0
print("\nThe steps are: ")
for i in data: 
    k=0
    for j in i:
        if hypo[k] != j:
            hypo[k] = '?'
        k= k+1
    print(hypo)

print("\n The final Answer is:")
print(hypo[:-1])