import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging

from trm_puzzle.dataset import ChessTRMDataset
from trm_puzzle.core import TinyRecursiveChessModel
from chess_seq.encoder import MoveEncoder


def run_overfit(
    max_epochs: int = 5000,
    print_every: int = 50,
    batch_size: int = 128,
    target_loss: float = 0.1,
):
    """
    Take one batch from the dataset, train on that single batch until
    the training loss <= `target_loss` (or `max_epochs` reached), and
    print the test loss (computed on a held-out batch if available).
    """
    data_path = "trm_puzzle/data/mate_in_1.npz"
    ds = ChessTRMDataset(data_path)

    move_encoder = MoveEncoder()
    move_encoder.load("data/move_encoder.pkl")
    num_moves = move_encoder.id_to_token.__len__()

    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
        )
    logger.info(
        "Loaded dataset %s: X.shape=%s X.max=%s num_moves=%d",
        data_path,
        ds.X.shape,
        int(ds.X.max()),
        num_moves,
    )

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    it = iter(loader)
    train_x, train_y = next(it)
    test_x, test_y = next(it)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_x = train_x.to(device)
    train_y = train_y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)

    logger.info(
        "Train batch: x=%s y=%s; Test batch: x=%s y=%s",
        tuple(train_x.shape),
        tuple(train_y.shape),
        tuple(test_x.shape),
        tuple(test_y.shape),
    )

    logger.info("token_vocab_size=%d", int(ds.X.max()) + 1)

    model = TinyRecursiveChessModel(
        token_vocab_size=int(ds.X.max()) + 1,
        num_moves=num_moves,
        dim=128,
        H_cycles=2,
        L_cycles=2,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

    emb_size = model.token_emb.num_embeddings
    logger.info(
        "Created model on %s: token_emb=%s head_out=%d",
        device,
        emb_size,
        model.head.out_features,
    )

    start = time.time()
    last_train_loss = float("inf")
    last_test_loss = float("inf")
    for epoch in range(1, max_epochs + 1):
        model.train()

        optim.zero_grad()
        logits = model(train_x)
        loss = F.cross_entropy(logits, train_y)
        loss.backward()
        optim.step()

        train_loss = loss.item()
        last_train_loss = train_loss

        # evaluate on test batch
        model.eval()
        with torch.no_grad():
            logits_t = model(test_x)
            test_loss = F.cross_entropy(logits_t, test_y).item()
            last_test_loss = test_loss
            preds = logits_t.argmax(dim=-1)
            test_acc = (preds == test_y).float().mean().item()

        if epoch % print_every == 0 or train_loss <= target_loss:
            elapsed = time.time() - start
            logger.info(
                "epoch=%d train_loss=%.6f test_loss=%.6f test_acc=%.4f elapsed=%.1fs",
                epoch,
                train_loss,
                test_loss,
                test_acc,
                elapsed,
            )

        if train_loss <= target_loss:
            logger.info("Reached train_loss=%.6f at epoch %d", train_loss, epoch)
            break

    total_time = time.time() - start
    logger.info(
        "Finished. Time: %.1fs, final_train_loss=%.6f, final_test_loss=%.6f, final_test_acc=%.4f",
        total_time,
        last_train_loss,
        last_test_loss,
        test_acc,
    )


if __name__ == "__main__":
    run_overfit()
