//! Streaming Percentile Calculator
//!
//! Project Context: Calculates percentiles from datasets too large to fit in memory
//! using a multi-pass binary search algorithm. Each pass reads through the data once
//! to count values relative to a threshold, requiring O(1) memory regardless of
//! dataset size.
//!
//! To perform binary search we need the range to search (min to max)
//! and total count to calculate percentages
//! for comparison (pct_below = count_below / total_count)
//!
//! Algorithm: Binary search over value range to find the value where the
//! percentage of values at or below it equals the target_percentile.
//! We compare (count_below / total_count) to target_percentile.
//!
//! Usage: User provides iterators over their data. This module provides the
//! calculation functions. Data source (files, database, etc.) is user's concern.

use std::fs::File;
use std::io::{self, BufRead, BufReader};

/*
Seven-Variable Binary-Search Method: For No-Load Percentile-Search

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
*/

/// Maximum iterations for binary search to prevent infinite loops
/// Calculation: log2((f64::MAX - f64::MIN) / f64::EPSILON) ≈ 64 iterations maximum
const MAX_ITERATIONS: usize = 100;

/// Errors during percentile calculation
/// Production error messages contain no sensitive information
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PercentileError {
    /// No data points in dataset
    EmptyDataset,
    /// Percentile must be between 0.0 and 1.0
    InvalidPercentile,
    /// Epsilon must be positive
    InvalidEpsilon,
    /// Binary search did not converge within max iterations
    NoConvergence,
}

/// Statistics from analyzing a dataset
///
/// Project Context: This is all we need to perform binary search -
/// the range to search (min to max) and total count to calculate percentages
/// for comparison (pct_below = count_below / total_count)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DatasetStats {
    pub min: f64,
    pub max: f64,
    pub count: usize,
}

/// Step 1: Scan through data once to get min, max, and count
///
/// Project Context: This is the first required pass. We need min and max
/// to define our search range, and count to calculate percentages
/// (pct_below = count_below / count) for comparison to target_percentile.
///
/// # Arguments
/// * `values` - Iterator providing f64 values (consumed by this function)
///
/// # Returns
/// * `Ok(DatasetStats)` - Statistics about the dataset
/// * `Err(PercentileError::EmptyDataset)` - If iterator yields no values
///
/// # Example
/// ```
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let stats = analyze(data.iter().copied())?;
/// assert_eq!(stats.min, 1.0);
/// assert_eq!(stats.max, 5.0);
/// assert_eq!(stats.count, 5);
/// ```
pub fn analyze<I>(values: I) -> Result<DatasetStats, PercentileError>
where
    I: Iterator<Item = f64>,
{
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut count: usize = 0;

    // Single pass through data
    for value in values {
        if value < min {
            min = value;
        }
        if value > max {
            max = value;
        }

        // Saturating add prevents overflow - if we hit usize::MAX, stay there
        // In practice, datasets this large would have other issues first
        count = count.saturating_add(1);
    }

    // Debug assertion for development
    debug_assert!(
        min <= max || count == 0,
        "Minimum must be <= maximum unless empty"
    );

    // Test assertion for test builds
    #[cfg(test)]
    if count > 0 {
        assert!(min.is_finite() && max.is_finite(), "Values must be finite");
    }

    // Production check: empty dataset
    if count == 0 {
        return Err(PercentileError::EmptyDataset);
    }

    // Production check: infinite or NaN values
    if !min.is_finite() || !max.is_finite() {
        return Err(PercentileError::EmptyDataset);
    }

    Ok(DatasetStats { min, max, count })
}

/// Step 2 helper: Count how many values are at or below threshold
///
/// Project Context: Called once per binary search iteration. By counting
/// how many values fall below our current guess, we determine whether to
/// search higher or lower.
///
/// # Arguments
/// * `values` - Iterator providing f64 values (consumed by this function)
/// * `threshold` - Value to compare against
///
/// # Returns
/// Count of values where value <= threshold
///
/// # Example
/// ```
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let count = count_below_threshold(data.iter().copied(), 3.0);
/// assert_eq!(count, 3); // 1.0, 2.0, 3.0 are <= 3.0
/// ```
pub fn count_below_threshold<I>(values: I, threshold: f64) -> usize
where
    I: Iterator<Item = f64>,
{
    let mut count: usize = 0;

    for value in values {
        if value <= threshold {
            // Saturating add prevents overflow
            count = count.saturating_add(1);
        }
    }

    count
}

/// Step 3: Binary search for percentile value
///
/// Project Context: This implements the core percentile algorithm using binary
/// search over the VALUE RANGE [min, max] (not over counts or indices).
///
/// Algorithm Logic:
/// 1. Start with search range: low=min, high=max
/// 2. Guess a value: mid = (low + high) / 2
/// 3. Count how many data points are <= mid
/// 4. Calculate: pct_below = (count_below / total_count)
/// 5. Compare pct_below to target_percentile:
///    - If pct_below < target_percentile: search HIGHER (low = mid)
///    - If pct_below >= target_percentile: search LOWER (high = mid)
/// 6. Repeat until range narrows below epsilon
///
/// Each iteration requires one full pass through the data to count values.
/// Number of iterations ≈ log₂((max-min)/epsilon)
///
/// Example: For [1,2,3,4,5], finding 50th percentile:
/// - Iteration 1: mid=3.0, count_below=3, pct_below=60% > 50% → search lower
/// - Iteration 2: mid=2.0, count_below=2, pct_below=40% < 50% → search higher
/// - Converges to value between 2.0 and 3.0
///
/// # Arguments
/// * `stats` - Dataset statistics from analyze()
/// * `target_percentile` - Desired percentile as fraction (0.0 to 1.0)
/// * `epsilon` - Convergence tolerance in value units (not percentage)
/// * `get_iterator` - Function that returns a fresh iterator over data each call
///
/// # Returns
/// * `Ok(f64)` - Estimated percentile value (not the count, the actual VALUE)
/// * `Err(PercentileError)` - If inputs invalid or convergence fails
///
/// # Example
/// ```
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let stats = analyze(data.iter().copied())?;
///
/// // Find the VALUE where 50% of data is below it
/// let p50 = binary_search_percentile(
///     &stats,
///     0.5,  // 50th percentile (median)
///     0.01, // stop when range < 0.01
///     || data.iter().copied()
/// )?;
/// // p50 will be approximately 2.5 (between 2nd and 3rd values)
/// ```
pub fn binary_search_percentile<F, I>(
    stats: &DatasetStats,
    target_percentile: f64,
    epsilon: f64,
    mut get_iterator: F,
) -> Result<f64, PercentileError>
where
    F: FnMut() -> I,
    I: Iterator<Item = f64>,
{
    // Validate inputs
    if !(0.0..=1.0).contains(&target_percentile) {
        return Err(PercentileError::InvalidPercentile);
    }
    if epsilon <= 0.0 || !epsilon.is_finite() {
        return Err(PercentileError::InvalidEpsilon);
    }

    debug_assert!(stats.min <= stats.max, "Min must be <= max");
    debug_assert!(stats.count > 0, "Count must be positive");

    #[cfg(test)]
    {
        assert!(stats.min.is_finite(), "Min must be finite");
        assert!(stats.max.is_finite(), "Max must be finite");
    }

    // Binary search bounds
    let mut low = stats.min;
    let mut high = stats.max;

    // Binary search loop with iteration limit
    for iteration in 0..MAX_ITERATIONS {
        let mid = (low + high) / 2.0;

        if !mid.is_finite() {
            return Err(PercentileError::NoConvergence);
        }

        // Check convergence
        if (high - low) <= epsilon {
            return Ok(mid);
        }

        // Count values below current guess
        let count_below = count_below_threshold(get_iterator(), mid);

        // Calculate percentage below (this is what we compare!)
        let pct_below = count_below as f64 / stats.count as f64;

        // Binary search: compare percentages
        if pct_below < target_percentile {
            // Too few values below mid, search higher
            low = mid;
        } else {
            // Too many values below mid, search lower
            high = mid;
        }

        #[cfg(debug_assertions)]
        if iteration % 10 == 0 {
            eprintln!(
                "DEBUG iteration {}: range [{}, {}], mid={}, count={}, pct={}",
                iteration, low, high, mid, count_below, pct_below
            );
        }
    }

    Err(PercentileError::NoConvergence)
}

/// Convenience function: Calculate percentile in one call
///
/// Project Context: Combines analyze() and binary_search_percentile() into
/// single function for common use case. Performs initial analysis pass,
/// then binary search passes.
///
/// # Arguments
/// * `target_percentile` - Desired percentile (0.0 to 1.0)
/// * `epsilon` - Accuracy tolerance
/// * `get_iterator` - Function returning fresh iterator each call
///
/// # Returns
/// * `Ok(f64)` - Estimated percentile value
/// * `Err(PercentileError)` - If inputs invalid or calculation fails
pub fn calculate_percentile<F, I>(
    target_percentile: f64,
    epsilon: f64,
    mut get_iterator: F,
) -> Result<f64, PercentileError>
where
    F: FnMut() -> I,
    I: Iterator<Item = f64>,
{
    // Step 1: Analyze dataset
    let stats = analyze(get_iterator())?;

    // Step 2: Binary search for percentile
    binary_search_percentile(&stats, target_percentile, epsilon, get_iterator)
}

// ============================================================================
// Example: Reading from a file
// ============================================================================

/// Example helper: Read f64 values from a text file (one per line)
///
/// Project Context: This is ONE way to provide data to the percentile calculator.
/// Users can implement their own data source (database, network, etc.)
///
/// # Arguments
/// * `path` - Path to file containing f64 values (one per line)
///
/// # Returns
/// Iterator over successfully parsed f64 values
///
/// # Production Safety
/// - Skips lines that fail to parse (no panic)
/// - Handles I/O errors gracefully
pub fn read_values_from_file(path: &str) -> io::Result<impl Iterator<Item = f64>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    Ok(reader.lines().filter_map(|line_result| {
        // Handle I/O error reading line
        let line = match line_result {
            Ok(l) => l,
            Err(_) => return None,
        };

        // Handle parse error
        match line.trim().parse::<f64>() {
            Ok(value) => Some(value),
            Err(_) => None,
        }
    }))
}

// ============================================================================
// Tests
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    /// Test: analyze() correctly finds min, max, and count
    ///
    /// Project Context: The analyze() function must accurately capture
    /// the data range and size for binary search to work. This verifies
    /// basic statistics extraction.
    ///
    /// Expected: For [1,2,3,4,5], min=1, max=5, count=5
    #[test]
    fn test_analyze_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = analyze(data.iter().copied()).unwrap();

        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.count, 5);
    }

    /// Test: analyze() returns error for empty dataset
    ///
    /// Project Context: Cannot calculate percentile of nothing.
    /// This ensures graceful error handling instead of panic.
    ///
    /// Expected: EmptyDataset error
    #[test]
    fn test_analyze_empty() {
        let data: Vec<f64> = vec![];
        let result = analyze(data.iter().copied());

        assert_eq!(result, Err(PercentileError::EmptyDataset));
    }

    /// Test: count_below_threshold() counts correctly
    ///
    /// Project Context: This is the core counting function used in
    /// each binary search iteration. Must count values <= threshold.
    ///
    /// Expected: For [1,2,3,4,5]:
    /// - threshold 0.0 → 0 values
    /// - threshold 3.0 → 3 values (1,2,3)
    /// - threshold 5.0 → 5 values (all)
    #[test]
    fn test_count_below_threshold() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert_eq!(count_below_threshold(data.iter().copied(), 0.0), 0);
        assert_eq!(count_below_threshold(data.iter().copied(), 3.0), 3);
        assert_eq!(count_below_threshold(data.iter().copied(), 5.0), 5);
        assert_eq!(count_below_threshold(data.iter().copied(), 10.0), 5);
    }

    /// Test: 50th percentile (median) calculation
    ///
    /// Project Context: For dataset [1,2,3,4,5]:
    /// - 50th percentile means 50% of values should be at or below
    /// - Algorithm compares: (count_below / 5) to 0.5
    /// - At threshold 2.0: pct_below = 2/5 = 40% < 50%
    /// - At threshold 3.0: pct_below = 3/5 = 60% > 50%
    /// - Algorithm converges between 2.0 and 3.0
    ///
    /// Note: For discrete data, percentile is approximate. Our binary
    /// search finds a value in the correct range.
    ///
    /// Expected: Result between 2.0 and 3.0 (reasonable median estimate)
    #[test]
    fn test_median() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = analyze(data.iter().copied()).unwrap();

        let median = binary_search_percentile(&stats, 0.5, 0.01, || data.iter().copied()).unwrap();

        // For 5 values, median should be between 2nd and 3rd value
        assert!(
            median >= 2.0 && median <= 3.0,
            "Median should be between 2.0 and 3.0, got {}",
            median
        );
    }

    /// Test: 25th and 75th percentiles (quartiles)
    ///
    /// Project Context: For dataset [1,2,3,4,5,6,7,8,9,10]:
    /// - Q1 (25th): Algorithm compares (count_below / 10) to 0.25
    ///   Should converge to value where ~2-3 values are below (20-30%)
    /// - Q3 (75th): Algorithm compares (count_below / 10) to 0.75
    ///   Should converge to value where ~7-8 values are below (70-80%)
    ///
    /// Note: With discrete data and binary search, exact values vary
    /// based on epsilon. We test for reasonable ranges.
    ///
    /// Expected: Q1 in [2,4], Q3 in [7,9]
    #[test]
    fn test_quartiles() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let stats = analyze(data.iter().copied()).unwrap();

        let q25 = binary_search_percentile(&stats, 0.25, 0.01, || data.iter().copied()).unwrap();

        let q75 = binary_search_percentile(&stats, 0.75, 0.01, || data.iter().copied()).unwrap();

        // Q1 should be in lower quartile range, Q3 in upper quartile range
        assert!(
            q25 >= 2.0 && q25 <= 4.0,
            "Q1 should be between 2.0 and 4.0, got {}",
            q25
        );
        assert!(
            q75 >= 7.0 && q75 <= 9.0,
            "Q3 should be between 7.0 and 9.0, got {}",
            q75
        );
    }

    /// Test: Invalid percentile values rejected
    ///
    /// Project Context: Percentile must be in [0.0, 1.0] range.
    /// Values outside this range are meaningless and should error
    /// instead of producing garbage results.
    ///
    /// Expected: InvalidPercentile error for values < 0 or > 1
    #[test]
    fn test_invalid_percentile() {
        let data = vec![1.0, 2.0, 3.0];
        let stats = analyze(data.iter().copied()).unwrap();

        assert_eq!(
            binary_search_percentile(&stats, -0.1, 0.01, || data.iter().copied()),
            Err(PercentileError::InvalidPercentile)
        );

        assert_eq!(
            binary_search_percentile(&stats, 1.5, 0.01, || data.iter().copied()),
            Err(PercentileError::InvalidPercentile)
        );
    }

    /// Test: Invalid epsilon values rejected
    ///
    /// Project Context: Epsilon controls convergence accuracy and must
    /// be positive. Zero or negative epsilon would cause infinite loops
    /// or invalid results.
    ///
    /// Expected: InvalidEpsilon error for epsilon <= 0
    #[test]
    fn test_invalid_epsilon() {
        let data = vec![1.0, 2.0, 3.0];
        let stats = analyze(data.iter().copied()).unwrap();

        assert_eq!(
            binary_search_percentile(&stats, 0.5, 0.0, || data.iter().copied()),
            Err(PercentileError::InvalidEpsilon)
        );

        assert_eq!(
            binary_search_percentile(&stats, 0.5, -0.01, || data.iter().copied()),
            Err(PercentileError::InvalidEpsilon)
        );
    }

    /// Test: Convenience function works end-to-end
    ///
    /// Project Context: calculate_percentile() combines analyze() and
    /// binary_search_percentile() in one call. This tests the complete
    /// workflow from raw data to percentile result.
    ///
    /// Expected: Median result in reasonable range
    #[test]
    fn test_convenience_function() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let median = calculate_percentile(0.5, 0.01, || data.iter().copied()).unwrap();

        // Median should be in middle range for this dataset
        assert!(
            median >= 2.0 && median <= 3.0,
            "Median should be between 2.0 and 3.0, got {}",
            median
        );
    }
}

fn main() {
    // Example 1: From Vec
    println!("=== Example 1: From Vec ===");
    let data = vec![1.0, 5.0, 3.0, 9.0, 7.0, 2.0, 8.0, 4.0, 6.0, 10.0];

    match calculate_percentile(0.50, 0.01, || data.iter().copied()) {
        Ok(p50) => println!("50th percentile: {}", p50),
        Err(e) => println!("Error: {:?}", e),
    }

    match calculate_percentile(0.75, 0.01, || data.iter().copied()) {
        Ok(p75) => println!("75th percentile: {}", p75),
        Err(e) => println!("Error: {:?}", e),
    }

    // Example 2: Step-by-step
    println!("\n=== Example 2: Step-by-step ===");

    match analyze(data.iter().copied()) {
        Ok(stats) => {
            println!(
                "Stats: min={}, max={}, count={}",
                stats.min, stats.max, stats.count
            );

            match binary_search_percentile(&stats, 0.90, 0.01, || data.iter().copied()) {
                Ok(p90) => println!("90th percentile: {}", p90),
                Err(e) => println!("Error: {:?}", e),
            }
        }
        Err(e) => println!("Analysis error: {:?}", e),
    }
}
