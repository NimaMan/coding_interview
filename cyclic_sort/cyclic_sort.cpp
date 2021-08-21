#include <iostream>
#include <vector>
using namespace std;

class CyclicSort {
 public:
  static void sort(vector<int> &nums) {
    // TODO: Write your code here    
    int i = 0;
    while (i< nums.size()){
      if (nums[i] == i+1){
        i++;
      }
      else{
        int this_num = nums[i]
        int other_num = nums[this_num -1];
        int this_num = nums[other_num_idx];
        nums[i] = other_num;
        nums[this_num - 1] = this_num;
    } 
  }
};
