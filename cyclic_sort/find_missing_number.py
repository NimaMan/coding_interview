"""
Simple way: first do a cyclic sort then look for the missing number 
Time Complexity: O(N) + O(N-1) + O(N)

Better way: track the number of swaps you do. At max you need n-i swaps to get the right number in the righjt index
O(N)

"""

def find_missing_number(nums):
  i = 0 
  num_swaps = 0  
  n = len(nums) 
  
  while num_swaps <= n - i: #O(N)
    j = nums[i] 
    #print(nums, j, i, num_swaps)
    if j == i: 
        i += 1 
        num_swaps = 0 
    elif j == n: 
        nums[i], nums[-1] = nums[-1], j  # swap 
    elif nums[i] != nums[j]:
        nums[i], nums[j] = nums[j], nums[i]  # swap
    if num_swaps == n-i:
        return i  
    num_swaps += 1
    


print(find_missing_number([4, 0, 3, 1]))
print(find_missing_number([8, 3, 5, 2, 4, 6, 0, 1]))


