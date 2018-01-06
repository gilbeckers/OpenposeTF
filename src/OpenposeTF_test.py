from . import OpenposeTF

def test_OpenposeTF():
    assert OpenposeTF.apply("Jane") == "hello Jane"
