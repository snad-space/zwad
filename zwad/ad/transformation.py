import numpy as np

ZERO_PERIOD_MASK = 1e-5

SUFFIXES_PASSBAND = ('_g', '_r', '_i')
SUFFIXES_FLUX_MAGN = ('_flux', '_magn')


def parse_feature_name(feature_name):
    """
    Normalize feature names.

    Removes "magn"/"flux" and passband (gri) suffixes and
    transforms to lower case

    Parameters
    ----------
    feature_name: file with features' names.

    Returns
    -------
    Feature names without suffixes and passbands.
    """
    feature_name = feature_name.lower()
    for suffix in SUFFIXES_PASSBAND:
        if feature_name.endswith(suffix):
            feature_name = feature_name[:-len(suffix)]
    for suffix in SUFFIXES_FLUX_MAGN:
        if feature_name.endswith(suffix):
            feature_name = feature_name[:-len(suffix)]
    return feature_name


def identical(x):
    return x


def period_norm(data):
    data = data.copy()
    data[data == 0] = ZERO_PERIOD_MASK
    return np.log10(data)


def period_norm_inv(data):
    data = data.copy()
    data[data == ZERO_PERIOD_MASK] = 0
    return np.power(10, data)


transform_direct = {'amplitude': identical,
                    'anderson_darling_normal': np.log10,
                    'beyond_1_std': identical,
                    'beyond_2_std': np.sqrt,
                    'chi2': np.log1p,
                    'cusum': identical,
                    'eta_e': np.log10,
                    'eta': identical,
                    'excess_variance': identical,
                    'inter_percentile_range_10': identical,
                    'inter_percentile_range_25': identical,
                    'inter_percentile_range_2': identical,
                    'kurtosis': np.arcsinh,
                    'linear_fit_reduced_chi2': np.log1p,
                    'linear_fit_slope': identical,
                    'linear_fit_slope_sigma': np.log10,
                    'linear_trend': identical,
                    'linear_trend_sigma': np.log10,
                    'magnitude_percentage_ratio_20_5': identical,
                    'magnitude_percentage_ratio_20_10': identical,
                    'magnitude_percentage_ratio_40_5': identical,
                    'maximum_slope': np.log10,
                    'mean': identical,
                    'mean_variance': identical,
                    'median_absolute_deviation': identical,
                    'median_buffer_range_percentage_5' : identical,
                    'median_buffer_range_percentage_10': identical,
                    'median_buffer_range_percentage_20': identical,
                    'percent_amplitude': identical,
                    'percent_difference_magnitude_percentile_20': np.log10,
                    'percent_difference_magnitude_percentile_10': np.log10,
                    'percent_difference_magnitude_percentile_5': np.log10,
                    'period_0': period_norm,
                    'period_1': period_norm,
                    'period_2': period_norm,
                    'period_3': period_norm,
                    'period_4': period_norm,
                    'period_s_to_n_0': np.arcsinh,
                    'period_s_to_n_1': np.arcsinh,
                    'period_s_to_n_2': np.arcsinh,
                    'period_s_to_n_3': np.arcsinh,
                    'period_s_to_n_4': np.arcsinh,
                    'periodogram_cusum': identical,
                    'periodogram_eta': identical,
                    'periodogram_amplitude': np.log1p,
                    'periodogram_beyond_1_std': identical,
                    'periodogram_beyond_2_std': identical,
                    'periodogram_beyond_3_std': identical,
                    'periodogram_standard_deviation': np.log1p,
                    'periodogram_inter_percentile_range_25': identical,
                    'periodogram_percent_amplitude': identical,
                    'skew': np.arcsinh,
                    'standard_deviation': identical,
                    'stetson_k': identical,
                    'weighted_mean': identical,
                    }

transform_inverse = {'amplitude': identical,
                     'anderson_darling_normal': lambda x: np.power(10, x),
                     'beyond_1_std': identical,
                     'beyond_2_std': lambda x: np.power(x, 2),
                     'chi2': np.expm1,
                     'cusum': identical,
                     'eta_e': lambda x: np.power(10, x),
                     'eta': identical,
                     'excess_variance': identical,
                     'inter_percentile_range_10': identical,
                     'inter_percentile_range_25': identical,
                     'inter_percentile_range_2': identical,
                     'kurtosis': np.sinh,
                     'linear_fit_reduced_chi2': np.expm1,
                     'linear_fit_slope': identical,
                     'linear_fit_slope_sigma': lambda x: np.power(10, x),
                     'linear_trend': identical,
                     'linear_trend_sigma': lambda x: np.power(10, x),
                     'magnitude_percentage_ratio_20_5': identical,
                     'magnitude_percentage_ratio_20_10': identical,
                     'magnitude_percentage_ratio_40_5': identical,
                     'maximum_slope': lambda x: np.power(10, x),
                     'mean': identical,
                     'mean_variance': identical,
                     'median_absolute_deviation': identical,
                     'median_buffer_range_percentage_5' : identical,
                     'median_buffer_range_percentage_10': identical,
                     'median_buffer_range_percentage_20': identical,
                     'percent_amplitude': identical,
                     'percent_difference_magnitude_percentile_20': lambda x: np.power(10, x),
                     'percent_difference_magnitude_percentile_10': lambda x: np.power(10, x),
                     'percent_difference_magnitude_percentile_5': lambda x: np.power(10, x),
                     'period_0': period_norm_inv,
                     'period_1': period_norm_inv,
                     'period_2': period_norm_inv,
                     'period_3': period_norm_inv,
                     'period_4': period_norm_inv,
                     'period_s_to_n_0': np.sinh,
                     'period_s_to_n_1': np.sinh,
                     'period_s_to_n_2': np.sinh,
                     'period_s_to_n_3': np.sinh,
                     'period_s_to_n_4': np.sinh,
                     'periodogram_cusum': identical,
                     'periodogram_eta': identical,
                     'periodogram_amplitude': np.expm1,
                     'periodogram_beyond_1_std': identical,
                     'periodogram_beyond_2_std': identical,
                     'periodogram_beyond_3_std': identical,
                     'periodogram_inter_percentile_range_25': identical,
                     'periodogram_standard_deviation': np.expm1,
                     'periodogram_percent_amplitude': identical,
                     'skew': np.sinh,
                     'standard_deviation': identical,
                     'stetson_k': identical,
                     'weighted_mean': identical,
                     }


def transform_features(values, feature_names):
    """
    Transform the table of features in-place using non linear functions
    from the dict `transform_direct`.

    Parameters
    ----------
    values: Matrix of N-by-M values with N experiments and M features.
    feature_names: file with features' names.

    Returns
    -------
    N-by-M matrix of transformed features.
    """
    if values.shape[1] != len(feature_names):
        raise ValueError('Length of feature_names is different from number of columns in values')
    feature_names = [parse_feature_name(name) for name in feature_names]
    for feature, column in zip(feature_names, values.T):
        func = transform_direct[feature]
        # We can make it faster using out=column in np functions
        column[:] = func(column)
    return values
