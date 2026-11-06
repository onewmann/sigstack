import sigstack


def test_deliberately_failing() -> None:
    # Deliberate failure to show that CI turns red; reverted in the next commit.
    assert sigstack.__version__ == "not-a-version"
