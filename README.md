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
approch --> one time include the ans and one time exclude the ans  
           when include the ans then call for the (index +1), when we include the ans than call for hte (ind+2)
    
### Recursive approch --- *Time Limit Exceeded*

```cpp
