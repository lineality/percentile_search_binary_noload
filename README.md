# percentile_search_binary_noload


```
7 variable binary search method:

1. epsilon: Epsilon controls accuracy: Smaller epsilon = more passes but more precise
2. current_observed_value_target
3. total_number_of_datapoints
4 min
5. max
6. number_of_values_below_target
7. % pct_of_values_below = (number_of_values_below_target/total_number_of_datapoints) * 100

Step 1. pick data, pick Epsilon
step 2. scan-pass through data once to get: total_number_of_datapoints, min, max
step 3. guess a percentile value based on min-max -> let mut current_observed_value
step 4. scan through data and get pct_of_values_below

step 5: Like 'binary search': adjust current_observed_value
- if too low increase current_observed_value and try again
- if too high decrease current_observed_value and try again

repeat until roughly at percentile... 
```


```
// Step 2: scan-pass to get total_number_of_datapoints, min, max
let stats = analyze(data)?;  // ✓

// Step 3: guess based on min-max
let mut low = stats.min;
let mut high = stats.max;
let mid = (low + high) / 2.0;  // ✓ current_observed_value

// Step 4: scan through data and get pct_of_values_below
let count_below = count_below_threshold(data, mid);  // ✓
let pct_below = count_below as f64 / stats.count as f64;  // ✓

// Step 5: binary search adjust
if pct_below < target_percentile {
    low = mid;  // ✓ too low, increase
} else {
    high = mid; // ✓ too high, decrease
}

// Repeat until roughly at percentile
while (high - low) > epsilon { ... }  // ✓
```
