from subprocess import run, Popen, PIPE


def test_zwadp():
    """
    Test run of zwadp
    """
    r = run(['zwadp',
             '--oid', 'data/fakes/oid_m31_fake.dat',
             '--feature', 'data/fakes/feature_m31_fake.dat',
             '-t', '--feature-names', 'data/fakes/feature_m31_fake.name'])
    assert r.returncode == 0


def test_zwann():
    """
    Test run of zwann
    """
    r = run(['zwann',
             '--oid', 'data/fakes/oid_m31_fake.dat',
             '--feature', 'data/fakes/feature_m31_fake.dat',
             '--lookup', '0'])
    assert r.returncode == 0

