#%%
# Counting the number of pairs with sum = 10
def countPair():            
    count = 0
    arr = [2, 7, 4, 1, 3, 6]
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            if (arr[i] + arr[j] == 10):
                count += 1
    return count

def main():
    x = countPair()
    print("No. of pairs equal to 10:", x)

main()

#%%
# Calculating the range of the given input list and showing error if length is less than 3
def retRange(arr):
    n = len(arr)
    if n < 3:
        return "Range is not possible"
    else:
        min_val = min(arr)
        max_val = max(arr)
        return (min_val, max_val)

def main():
    n = input("Enter the elements: ")
    arr = list(map(int, n.strip().split()))
    x = retRange(arr)
    print("Range of List is", x)

main()
# %%
# Returning the matrix multiplied by m times
def matrix(A, m):
    n = len(A)

    def matrix_matrix(p, q):
        result = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    result[i][j] += p[i][k] * q[k][j]
        return result

    result = A
    for _ in range(m - 1):
        result = matrix_matrix(result, A)
    return result

def main():
    n = int(input("Enter the size of the matrix: "))
    print("Enter the elements of the matrix:")
    A = []
    for _ in range(n):
        arr = list(map(int, input().split()))
        A.append(arr)

    m = int(input("Enter the positive number: "))
    if m < 1:
        print("Not positive")
        return

    x = matrix(A, m)

    print("Resultant Matrix:")
    for r in x:
        print(r)

main()
#%%
# Returning the highest occurrence char in a string with frequency
def highOccurnce(s):
    count = {}

    for i in s:
        if i != ' ':
            count[i] = count.get(i, 0) + 1

    character = max(count, key=count.get)
    max_count = count[character]

    return character, max_count

def main():
    s = input("Enter the String: ")
    char, count = highOccurnce(s)
    print(f"Highest occurrence character is '{char}' with count {count}")

main()
#%%
# Returning the mode, mean, and median of the randomly generated 25 numbers in range 1 to 10
import random
import statistics

random_List = [random.randint(1, 10) for _ in range(25)]
print("List:", random_List)

value_mean = statistics.mean(random_List)
value_median = statistics.median(random_List)

try:
    value_mode = statistics.mode(random_List)
except statistics.StatisticsError:
    value_mode = "No unique mode"

print("Mean:", value_mean)
print("Median:", value_median)
print("Mode:", value_mode)
