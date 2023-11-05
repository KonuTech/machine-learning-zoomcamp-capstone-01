def calculate_correlation_with_target(features_df, target_series, variance_threshold=0.0):
    """
    Calculate the correlation of numeric columns in a features DataFrame with a target Series
    and perform Variance Threshold feature selection.
    
    Parameters:
    features_df (pd.DataFrame): The features DataFrame.
    target_series (pd.Series): The target Series.
    variance_threshold (float): Variance threshold for feature selection. Features with variance
        below this threshold will be removed. Default is 0.0 (no threshold).

    Returns:
    pd.Series: A Series containing the correlation coefficients sorted by absolute values.
    """
    # Select only numeric columns from the features DataFrame
    numeric_features = features_df.select_dtypes(include=['number'])
    
    # Calculate the correlation and sort the result by absolute values in descending order
    correlation_series = numeric_features.corrwith(target_series)
    absolute_correlation_series = correlation_series.abs()
    
    # Apply Variance Threshold to filter features
    if variance_threshold > 0.0:
        selector = VarianceThreshold(threshold=variance_threshold)
        numeric_features = selector.fit_transform(numeric_features)
        # Update correlation series to match the selected features
        correlation_series = pd.Series(selector.inverse_transform(correlation_series.values.reshape(1, -1))[0], index=numeric_features.columns)
    
    # Sort the DataFrame by absolute values
    correlation_series = correlation_series.sort_values(ascending=False)
    
    return correlation_series
