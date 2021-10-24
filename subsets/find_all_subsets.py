"""
Find all subsets of a set of numbers: A recursive approach
    - find

"""


def find_all_subsets_recursive(nums):
    subsets = []
    def find_subsets(nums):
        print(nums)
        if len(nums) == 1:
            subsets.append([])
            subsets.append(nums)
            return [[], nums]
        else:
            new_subsets = find_subsets(nums[:-1])
            print(new_subsets)
            i = 0
            for nsb in new_subsets:
                subsets.append([nums[-1]] + nsb)
                print(new_subsets)
                i+=1
                if i==7:
                    break
            return subsets

    return find_subsets(nums)


def find_all_subsets(nums):
    subsets = [[]]
    for num in nums:
        new_subsets = []
        for sb in subsets:
            new_subsets.append([num] + sb)
        subsets = subsets + new_subsets
    return subsets


find_all_subsets([1, 3])
find_all_subsets_recursive([1, 5, 3])