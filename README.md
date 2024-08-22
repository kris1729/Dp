# DP
> - learn from past ,  no reapeat calculation store ans ans use
> - dp is your friend for DREAM Company

## Approch to solve 
>- 1. apply recursion , apply top down , apply bottom up , space optimization

- ### top down -> memoization + recursion
- ### bottom up -> Tabulation 

# when We Apply Dp
- Overlapaing sub problem 
- Optimal sub Problem
---
# Fibonacci Number 
![](./img/fibonacci%20Number.png)

code using recursion
```cpp
int fib(int n) {
        if (n <= 1)
            return n;
        int ans = fib(n - 1) + fib(n - 2);
        return ans;
    }
```
code using top down approch

```cpp
  int solve(vector<int>& dp, int n) {
        if (n <= 1)
            return n;
        if (dp[n] != -1)
            return dp[n];
        return dp[n] = solve(dp, n - 1) + solve(dp, n - 2);
    }
    int fib(int n) {
        vector<int> dp(n + 1, -1);
        return solve(dp, n);
    }
```
code using bottom up approch

```cpp
  int fib(int n) {
        vector<int> dp(n + 2, -1); // at n =0 give erroe dp[1] =1;
        dp[0] = 0, dp[1] = 1;
        for (int i = 2; i <= n; i++)
            dp[i] = dp[i - 1] + dp[i - 2];
        return dp[n];
    }
```
code using space optimization

```cpp
int fib(int n) {
        int pre = 0, cur = 1;
        if (n <= 1)
            return n;
        for (int i = 2; i <= n; i++) {
            int ans = pre + cur;
            pre = cur;
            cur = ans;
        }
        return cur;
    }
```
> # Climbing Stairs
![](./img/Climbing%20Stairs.png)

it is like fibo serise 1,2,3,5,...

> Recursion code *Time LIMT EXeed* 
```cpp
 int solve(int n) {
        if (n <= 2)
            return n;
        int ans = solve(n - 1) + solve(n - 2);
        return ans;
    }
    int climbStairs(int n) {
        int ans = solve(n);
        return ans;
    }
```
> using memozation , top down approch
```cpp
int solve(int n,vector<int>&dp) {
        if (n <= 2)
            return n;
            if(dp[n]!=-1)return dp[n];
        return dp[n] = solve(n - 1,dp) + solve(n - 2,dp);
    }
    int climbStairs(int n) {
        vector<int>dp(n+1,-1);
        int ans = solve(n,dp);
        return ans;
    }
```
> using bottom up approch (tabulation )
```cpp
 int climbStairs(int n) {
        vector<int> dp(n + 1, -1);
        dp[0] = 1, dp[1] = 2;
        for (int i = 2; i < n; i++)
            dp[i] = dp[i - 2] + dp[i - 1];
        return dp[n - 1];
    }
```
> using most optimize approch *best solution*
```cpp
 int climbStairs(int n) {
        if (n <= 2)
            return n;
            
        int pre1 = 1;
        int pre = 2;
        int cur = 0;
        for (int i = 2; i < n; i++) {
            cur = pre1 + pre;
            pre1 = pre;
            pre = cur;
        }
        return cur;
    }
```
> # Min Cost Climbing Stairs
![](./img/Min%20Cost%20Climbing%20Stairs.png)

> using recursion approch wiil give *TLE*
```cpp
 int solve(vector<int>& cost, int i, int n) {
        if (i >= n) return 0;
        int ans = cost[i];
        ans += min(solve(cost, i + 2, n), solve(cost, i + 1, n));
        return ans;
    }
    int minCostClimbingStairs(vector<int>& cost) {
        int n = cost.size();
        int ans1 = solve(cost, 0, n);
        int ans2 = solve(cost, 1, n);
        return min(ans1, ans2);
    }
```
> using top down approch
```cpp
 int solve(vector<int>& cost, int i, int n, vector<int>& dp) {
        if (i >= n) return 0;
        if (dp[i] != -1) return dp[i];
        int ans = cost[i];
        ans += min(solve(cost, i + 2, n, dp), solve(cost, i + 1, n, dp));
        return dp[i] = ans;
    }
    int minCostClimbingStairs(vector<int>& cost) {
        int n = cost.size();
        vector<int> dp(n + 1, -1);
        int ans1 = solve(cost, 0, n, dp);
        int ans2 = solve(cost, 1, n, dp);
        return min(ans1, ans2);
    }
```
> bottom up approch
```cpp
 int minCostClimbingStairs(vector<int>& cost) {
        int n = cost.size();
        vector<int> dp(n + 1, -1);
        dp[0] = cost[0], dp[1] = cost[1];

        for (int i = 2; i < n; i++) {
            dp[i] = cost[i] + min(dp[i - 1], dp[i - 2]);
        }
        return min(dp[n - 1], dp[n - 2]);
    }
```
> using space optimization *best solution*
```cpp
 int minCostClimbingStairs(vector<int>& cost) {
        int n = cost.size();
        vector<int> dp(n + 1, -1);
        int pre2 = cost[0], pre = cost[1], cur;

        for (int i = 2; i < n; i++) {
            cur = cost[i] + min(pre, pre2);
            pre2 = pre;
            pre = cur;
        }
        return min(pre2, pre);
    }
```
> # Frog Jump
![](./img/Frog%20Jump-1.png)
![](./img/Frog%20Jump-2.png)
approch , at any index first we will take the abs(pre-cur) and , solve for two case one for (i+1)th jump and other for
(i+2)Th jump

> recrusive solution *TLE*
```cpp
int solve(int i , int n , vector<int>&height,vector<int>&dp){
  if(i>=n-1)return abs(height[n-1]-height[i]);
  if(dp[i]!=-1)return dp[i];
  int ans = abs(height[i+2]-height[i]) + solve(i+2,n,height,dp);
  int ans1 = abs(height[i+1]-height[i])+ solve(i+1,n,height,dp);
  return dp[i] =  min(ans ,ans1);
} 
int frogJump(int n, vector<int> &heights)
{
  vector<int>dp(n+1,-1);
 return  solve(0,n,heights,dp);
   
}
```
> Top down Approch  convert the recursion code into top down

```cpp
int solve(int i , int n , vector<int>&height,vector<int>&dp){
  if(i>=n-1)return abs(height[n-1]-height[i]);
  if(dp[i]!=-1)return dp[i];
  int ans = abs(height[i+2]-height[i]) + solve(i+2,n,height,dp);
  int ans1 = abs(height[i+1]-height[i])+ solve(i+1,n,height,dp);
  return dp[i] =  min(ans ,ans1);
} 
int frogJump(int n, vector<int> &heights)
{
  vector<int>dp(n+1,-1);
 return  solve(0,n,heights,dp);
   
}
```
> using bottom up approch 
```cpp
int frogJump(int n, vector<int> &arr) {
  vector<int> dp(n + 1, -1);
  dp[0] = 0 ,dp[1] =  abs(arr[0]-arr[1]);
    for(int i =2;i<n;i++){
      int ans1  = abs(arr[i]-arr[i-1]) + dp[i-1];
      int ans2 = abs(arr[i]-arr[i-2]) + dp[i-2];
      dp[i] = min(ans1 ,ans2); 
    }
    return dp[n-1];
}
```
> Space optimize *best solution*
```cpp
int frogJump(int n, vector<int> &arr) {
  
  int pre1 = 0 , pre=  abs(arr[0]-arr[1]);
    for(int i =2;i<n;i++){
      int ans1  = abs(arr[i]-arr[i-1]) + pre;
      int ans2 = abs(arr[i]-arr[i-2]) + pre1;
     int cur = min(ans1 ,ans2); 
     pre1 = pre;
     pre = cur;
    }
    return pre;
}
```


> # Coin Change # important problem
![](./img/Coin%20Change.png)
approch, we will try to find the spacific amount of monney using all possible coins , 
we does not use the greedy approch because at some case it give incruct answer

>  recursion solution --- *Time Limit Exceeded*
```cpp
int solve(vector<int>& coins, int n) {
        // base case
        if (n == 0) return 0;
        if (n < 0)return INT_MAX;
        // main case
        int mini = INT_MAX;
        for (auto x : coins) {
            int ans = solve(coins, n - x);
            if (ans != INT_MAX)
                mini = min(mini, ans + 1);
        }
        return mini;
    }
    int coinChange(vector<int>& coins, int amount) {
        int ans = solve(coins, amount);
        return ans == INT_MAX ? -1 : ans;
    }
```
>   Top down approch ----  *converted the recursion code int dp*

```cpp
int solve(vector<int>& coins, vector<int>& dp, int n) {
        // base case
        if (n == 0)return 0;
        if (n < 0) return INT_MAX;

        // check dp contain solution
        if (dp[n] != -1)  return dp[n];

        // store the solution in dp
        int mini = INT_MAX;
        for (auto x : coins) {
            int ans = solve(coins, dp, n - x);
            if (ans != INT_MAX)
                mini = min(mini, ans + 1);
        }
        return dp[n] = mini;
    }
    int coinChange(vector<int>& coins, int amount) {
        vector<int> dp(amount + 1, -1);
        
        int ans = solve(coins, dp, amount);
        return ans == INT_MAX ? -1 : ans;
    }
```
> Bottom up approch  *important apprech*
 ```cpp
 int coinChange(vector<int>& coins, int amount) {
        int n = amount;
        // create dp
        vector<int> dp(n + 1, -1);
        dp[0] = 0;

        // tabulation
        for (int i = 1; i <= n; i++) {
            int mini = INT_MAX;
            for (auto x : coins)
                if (i >= x && dp[i - x] != INT_MAX)
                    mini = min(mini, dp[i - x] + 1);
            dp[i] = mini;
        }
        return dp[n] == INT_MAX ? -1 : dp[n];
    }
```
> #  House Robber 
![](./img/House%20Robber.png)
approch --> one time include the ans and one time exclude the ans  
           when include the ans then call for the (index +1), when we include the ans than call for hte (ind+2)
    
>  Recursive approch --- *Time Limit Exceeded*

```cpp
int solve(int n, int index, vector<int>& nums) {
        if (index >= n) return 0;
        // include
        int inc = nums[index] + solve(n, index + 2, nums);
        // exclude
        int exc = solve(n, index + 1, nums);
        return max(inc, exc);
    }
    int rob(vector<int>& nums) {
        int n = nums.size();
        return solve(n, 0, nums);
    }
```
> Top down approch
```cpp
 int solve(int n, int index, vector<int>& nums, vector<int>& dp) {
        if (index >= n) return 0;
        if (dp[index] != -1)return dp[index];
        // include
        int inc = nums[index] + solve(n, index + 2, nums, dp);
        // exclude
        int exc = solve(n, index + 1, nums, dp);
        return dp[index] = max(inc, exc);
    }
    int rob(vector<int>& nums) {
        int n = nums.size();
        vector<int> dp(n + 1, -1);
        return solve(n, 0, nums, dp);
    }
```
> Bottom down approch --*Good approch*

```cpp
int rob(vector<int>& nums) {
        int n = nums.size();
        vector<int> dp(n + 1, -1);

        dp[0] = nums[0];
        if (n > 1) { // when n ==1 give error for excesing the nums[1]
            dp[1] = max(nums[0], nums[1]);
            for (int i = 2; i < n; i++) {
                int inc = nums[i] + dp[i - 2];
                int exc = dp[i - 1];
                dp[i] = max(inc, exc);
            }
        }
        return dp[n - 1];
    }
    //or 
/*

int rob(vector<int>& nums) {
        int n = nums.size();
        vector<int> dp(n + 1, -1);

        dp[0] = nums[0];
        for (int i = 1; i < n; i++) {
            int inc = nums[i];
            if (i > 1) inc += dp[i - 2];

            int exc = dp[i - 1];

            dp[i] = max(inc, exc);
        }

        return dp[n - 1];
    }

    */
```
>  Space Optimization *IMPORTANT*


for all the cur element we requried only dp[i-1] and dp[i-2] menas two variable requred 
```cpp
 int rob(vector<int>& nums) {
        // . . . . pre2 pre cur . . . . for any cur = max(pre , pre1) only two prev variable required
        int pre = nums[0];
        int pre2 = 0;
        int cur = nums[0];

        for (int i = 1; i < nums.size(); i++) {
            int take = nums[i] + pre2;
            int notTake = pre; // 0 + pre
            cur = max(take, notTake);
            pre2 = pre;
            pre = cur;
        }
        return cur;
    }
```
> # House Robber II
![](./img/House%20Robber%20II.png)

approch --> in b/w we calculate the ans one time from index 1 to last and one time 
from index 0 to last -1;
```cpp
// solve robber house -1 problem
  int solve(vector<int>& nums1) {
        int pre = nums1[0];
        int pre2 = 0;
        int cur = nums1[0];

        for (int i = 1; i < nums1.size(); i++) {
            int take = nums1[i] + pre2;
            int notTake = pre; // 0 + pre
            cur = max(take, notTake);
            pre2 = pre;
            pre = cur;
        }

        return cur;
    }
    int rob(vector<int>& nums) {
        // first and last will not take togater
        if (nums.size() > 1) {
            vector<int> num1(nums.begin() + 1, nums.end()); // excepts  nums[0] element
            vector<int> num2(nums.begin(), nums.end() - 1); // except nums[n-1] element

            int ans1 = solve(num1);
            int ans2 = solve(num2);
            return max(ans1, ans2);
        }
        return nums[0];
    }
```
> most space optimize approch
```cpp
 int solve(vector<int>& nums1, int st, int end) {
        int pre = nums1[st];
        int pre2 = 0;
        int cur = nums1[st];

        for (int i = st + 1; i < end; i++) {
            int take = nums1[i] + pre2;
            int notTake = pre; // 0 + pre
            cur = max(take, notTake);
            pre2 = pre;
            pre = cur;
        }

        return cur;
    }
    int rob(vector<int>& nums) {
        // first and last will not take togater
        int n = nums.size();
        if (nums.size() > 1) {
            int ans1 = solve(nums, 0, n - 1);
            int ans2 = solve(nums, 1, n);
            return max(ans1, ans2);
        }
        return nums[0];
    }
```

> # Painting the Fence 
![](./img/Painting%20the%20Fence.png)

approch ---- when 2 adsent is same + when 2 adsent is diff..


n =0 way is 0

n = 1             |   n = 2     |   n = 3  |     n = 4
                  |             |          |
same-  0          |  k          |   t1*k-1 |    t2*k-1
same-  k          |  t1*(k-1)   |   t2*k-1 |    t3*k-1
                  |             |          |
total  k          |   t2        |   t3     |     t4

> Recursion solution *TLE*
```cpp
long long solve(int n, int k)
{
    
    if (n == 1)
        return k;
    if (n == 2)
        return k + k * (k - 1) ; // k*K 
    int same = solve(n - 2, k) * (k - 1); // t1*(k-1)
    int diff = solve(n - 1, k) * (k - 1);  // t2*(k-1)
    return (same + diff);
}

    long long countWays(int n, int k){
        
        long long int ans = solve(n,k);
        return ans;
    }
```
> TOP down approch 
note - in the solve function send the value of n and k in long long form

```cpp
int MOD = 1e9 + 7;
// solve function
long long solve(long long  n, long long   k,vector<long long >&dp)
{
    
    if (n == 1)
        return k%MOD;
    if (n == 2)
        return (k + k * (k - 1))%MOD; 
    if(dp[n]!=-1)return dp[n];
    long long  same = (solve(n - 2, k,dp) * (k - 1))%MOD;
    long long  diff = (solve(n - 1, k,dp) * (k - 1))%MOD;
    return dp[n] = (same + diff)%MOD;
}
// main function 
    long long countWays(int n, int k){
        vector<long long > dp(n+1,-1);
        long long  ans = solve(n,k,dp);
        return ans;
}

```
> Bottom UP APPROCH
```cpp
 long long countWays(int n, int k){
       vector<long long > dp(n+1,-1);
       long long  K =k;
       int mod = 1e9+7;
       dp[1] = K%mod;
       dp[2] = K*K%mod;
       for(int i =3 ; i<=n;i++){
           dp[i] = ((K-1)*(dp[i-2]+dp[i-1]))%mod;
       }
       return dp[n]; 
    }
```
> space optimize approch
```cpp
 long long countWays(int n, int k){
       vector<long long > dp(n+1,-1);
       long long  K =k;
       int mod = 1e9+7;
       long long pre2 = K%mod;
       long long pre = K*K%mod;
       for(int i =3 ; i<=n;i++){
           long long cur = ((K-1)*(pre2+pre))%mod;
           pre2 = pre;
           pre = cur;
       }
       return pre; 
    }
```

> # Ninja’s Training
![](./img/Ninja’s%20Training-1.png)
![](./img/Ninja’s%20Training-2.png)
 its is  2d dp problem 
 Approch - > each time we select one , if select 0 then next time we will select 1 or 2 which give the max ans
 
 > solve using recursion *TLE*
 ```cpp
 int solve(int ind , int n , int day ,vector<vector<int>> &points){
   if(day>=n) return 0;
   int ans = points[day][ind];
   int ind1 = (ind+1)%3; // if {ind = 0 than ind1 =1,ind2 =2 } if {ind = 1 than ind1 =2,ind2 = 0} 
   int ind2 = (ind+2)%3; // if(ind = 2 than ind1 =1 , ind2 = 0)
   ans += max(solve(ind1 , n , day+1,points) , solve(ind2, n,day+1,points));
   return ans;
}
int ninjaTraining(int n, vector<vector<int>> &points)
{
   int ans = solve(0,n,0,points);
    int ans1 = solve(1,n,0,points);
    int ans2 = solve(2,n,0,points);
return max(ans,max(ans1,ans2));
}
```
> solve using Top down approch


change recursion code into top down approch
```cpp

int solve(int ind , int n , int day ,vector<vector<int>> &points,   vector<vector<int>>&dp){
    if(day>=n) return 0;
    if(dp[day][ind]!=-1)return dp[day][ind];
   int ans = points[day][ind];
   int ind1 = (ind+1)%3;
   int ind2 = (ind+2)%3;
   ans += max(solve(ind1 , n , day+1,points,dp) , solve(ind2, n,day+1,points,dp));
   return dp[day][ind]=ans;
}
int ninjaTraining(int n, vector<vector<int>> &points)
{
    vector<vector<int>>dp(n,vector<int>(3,-1));
   int ans = solve(0,n,0,points,dp);
    int ans1 = solve(1,n,0,points,dp);
    int ans2 = solve(2,n,0,points,dp);
return max(ans,max(ans1,ans2));
}
```
> solve using Bottom up approch *IMPORTANT*
```cpp
int ninjaTraining(int n, vector<vector<int>> &points)
{
   vector<vector<int>>dp(n,vector<int>(3,-1));
   dp[0][0] = points[0][0],dp[0][1] = points[0][1],dp[0][2] = points[0][2];

   for(int i =1;i<n;i++){
      
       dp[i][0] = points[i][0] + max(dp[i-1][1],dp[i-1][2]); // select first training
       dp[i][1] = points[i][1] + max(dp[i-1][0],dp[i-1][2]); // select second training
       dp[i][2] = points[i][2] + max(dp[i-1][0],dp[i-1][1]); // select third training
   }
 return max(dp[n-1][0],max(dp[n-1][1],dp[n-1][2]));
}
```
> ### space optimize *best solution*

```cpp
int ninjaTraining(int n, vector<vector<int>> &points)
{
    int pre0 = points[0][0],pre1 = points[0][1] , pre2 = points[0][2];
    for(int i =1;i<n;i++){
        int cur0 = points[i][0] + max(pre1,pre2);
        int cur1 = points[i][1] + max(pre0,pre2);
        int cur2 = points[i][2] + max(pre0,pre1);
        pre0 = cur0;
        pre1 = cur1;
        pre2 = cur2;
    }
    return max(pre0,max(pre1,pre2));   
}
```