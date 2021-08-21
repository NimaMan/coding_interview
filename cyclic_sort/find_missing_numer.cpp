using namespace std;

#include <iostream>
#include <vector>

void swap(vector<int>& arr, int i, int j) {
    int temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
  }

int findMissingNumber(vector<int>& nums) {
    int i = 0;
    int numSwaps = 0;
    int n = nums.size();

    while (numSwaps <= n - i) {
      int j = nums[i];  
      if (j == i) {
          i++;
          numSwaps = 0;
      } else if (j == n){
        
        swap(nums, i, j-1);
      }
      else {
        swap(nums, i, j);
      }
      if (numSwaps == n-i){
        return i;

      }
      numSwaps++;
    }
}


int main(int argc, char *argv[]) {
  vector<int> v1 = {4, 0, 3, 1};
  cout << findMissingNumber(v1) << endl;
  v1 = {8, 3, 5, 2, 4, 6, 0, 1};
  cout << findMissingNumber(v1) << endl;
}
