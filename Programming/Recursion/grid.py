''' recursion, for factorials '''

# grid_paths(n, m) -> 1 if n = 1 or m = 1

def grid_paths(n, m):
    if n == 1 or m == 1:
        return 1
    else:
        return grid_paths(n, m-1) + grid_paths(n - 1, m)

if __name__ == "__main__":

    print(grid_paths(2, 3))
