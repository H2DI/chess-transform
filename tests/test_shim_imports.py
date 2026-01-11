def test_top_level_imports():
    import chess_seq

    # These imports should succeed from the top-level shim
    from chess_seq import (
        MoveEncoder,
        ChessNet,
        ModelConfig,
        ChessGameEngine,
        ChessDataset,
        build_dataloader,
        build_and_save_model,
        load_model_from_checkpoint,
        clone_model,
        get_latest_checkpoint,
    )

    # Minimal validation of types
    assert callable(MoveEncoder)
    assert callable(ChessNet)
    assert callable(ModelConfig)
    assert callable(ChessGameEngine)
    assert callable(ChessDataset)
    assert callable(build_dataloader)
    assert callable(build_and_save_model)
    assert callable(load_model_from_checkpoint)
    assert callable(clone_model)
    assert callable(get_latest_checkpoint)
