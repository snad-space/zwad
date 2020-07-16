from subprocess import run, Popen, PIPE


def test_zwadp():
    """
    Test run of zwadp
    """
    r = run(['zwadp',
             '--oid', 'data/fakes/oid_m31_fake.dat',
             '--feature', 'data/fakes/feature_m31_fake.dat'])
    assert r.returncode == 0


def test_zwaad():
    """
    Test run of zwaad
    """
    p1 = Popen(['yes'], stdout=PIPE)
    p2 = Popen(['zwaad',
                '--oid', 'data/fakes/oid_m31_fake.dat',
                '--feature', 'data/fakes/feature_m31_fake.dat',
                '--budget', '5'], stdin=p1.stdout)
    assert p2.wait() == 0


def test_zwann():
    """
    Test run of zwann
    """
    r = run(['zwann',
             '--oid', 'data/fakes/oid_m31_fake.dat',
             '--feature', 'data/fakes/feature_m31_fake.dat',
             '--lookup', '0'])
    assert r.returncode == 0
