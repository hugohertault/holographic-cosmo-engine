def test_imports():
    import hce
    from hce.config import TrainConfig
    assert hasattr(hce, "__version__")
    assert TrainConfig().n_colloc > 0
