"""
Given a set of numbers in random order in range[1, n] sort 
O(N) time complexity and O(1) Space Complexity
"""


def cyclic_sort(nums):
	i = 0
  	while i < len(nums):
		if nums[i] == i+1:
		  	i += 1
	  	else:
		  	num_idx = nums[i] - 1
		 	nums[num_idx], nums[i] = nums[i], nums[num_idx]
  	return nums


if __name__ == "__main__":
	print(cyclic_sort([3, 1, 5, 4, 2]))