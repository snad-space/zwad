from zwad.ad.transformation import *


def test_transform_features():
    values = np.array([[1.2, 3.4, 5.8],
                       [1.18, 44.5, 12],
                       [0.45, 16.2, 2]])
    names = ['amplitude_flux', 'cusum', 'skew_magn_r']
    desired = transform_features(values, names)
    actual = np.array([[1.2, 3.4, 2.458355],
                       [1.18, 44.5, 3.179785],
                       [0.45, 16.2, 1.4436]])
    np.testing.assert_allclose(actual, desired, rtol=1e-4)


def test_transform_dicts_keys():
    assert set(transform_direct) == set(transform_inverse)


def test_direct_inverse_composition():
    x = np.linspace(0.5, 3.5)
    for feature in transform_direct:
        y = transform_direct[feature](x)
        np.testing.assert_allclose(x, transform_inverse[feature](y), rtol=1e-10)
