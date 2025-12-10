"""Streamlit UI to explore per-layer attention for a chosen piece and game.

Run with:

    pip install streamlit
    streamlit run scripts/visualize_ui.py

This app loads the same model/encoder used by the scripts and renders
the per-layer, square attention matrices with vertical+horizontal lines
marking the plies where the chosen piece moved.
"""

import io
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import torch
import chess
import chess.svg

try:
    import cairosvg

    _HAVE_CAIROSVG = True
except Exception:
    _HAVE_CAIROSVG = False


@st.cache_resource
def load_visualize_module(path: str = "scripts/visualize_knight_attention.py"):
    spec = importlib.util.spec_from_file_location("vk", path)
    vk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vk)
    return vk


@st.cache_resource
def load_model_and_encoder(model_name, special_name, device="cpu"):
    vk = load_visualize_module()
    model, model_config, _ = vk.load_model(
        model_name=model_name, special_name=special_name
    )
    model = model.to(device)
    encoder = vk.MoveEncoder()
    encoder.load(model_config.encoder_path)
    return model, encoder


def plot_per_layer_square(mats, move_labels, token_positions, layers, head):
    # mats: list of (T,T) numpy arrays, one per layer
    n_layers = len(mats)
    T = mats[0].shape[0]
    pixels_per_token = 0.04
    square_size = max(6, min(20, T * pixels_per_token))
    fig_w = square_size
    fig_h = square_size * n_layers
    fig, axes = plt.subplots(n_layers, 1, figsize=(fig_w, fig_h), squeeze=False)

    for i, mat in enumerate(mats):
        ax = axes[i, 0]
        vmax = float(max(1e-9, mat.max()))
        im = ax.imshow(
            mat,
            aspect="equal",
            origin="lower",
            extent=[0, T, 0, T],
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_title(f"Layer {layers[i]} | Head {head}")
        for tok_idx in token_positions:
            ax.axvline(tok_idx, color="white", linewidth=1.0, linestyle="--")
            ax.axhline(tok_idx, color="white", linewidth=1.0, linestyle="-")

        step = max(1, T // 32)
        xticks = list(range(0, T, step))
        xtick_labels = [move_labels[j] for j in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, rotation=90, fontsize=6)
        ax.set_yticks(xticks)
        ax.set_yticklabels(xtick_labels, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    fig.tight_layout()
    return fig


def main():
    st.title("Chess attention explorer")
    vk = load_visualize_module()

    with st.sidebar:
        pgn = st.text_input("PGN path", "data/gms_pgns/Ivanchuk.pgn")
        game_index = st.number_input("Game index", min_value=0, value=0)
        model_name = st.text_input("Model name", "gamba_rossa")
        special_name = st.text_input("Checkpoint name", "final")
        device = st.selectbox("Device", ["cpu", "cuda"], index=0)
        piece = st.text_input("Piece initial square", "g1")
        layers = st.text_input("Layers (comma)", "0,-1")
        head = st.number_input("Head (single)", min_value=0, value=0)
        max_tokens = st.number_input("Max tokens", min_value=8, value=256)
        single_plot = st.checkbox("Single per-layer plot", value=True)
        compute_btn = st.button("Compute attention (runs model)")
        plot_btn = st.button("Plot (use cached attention)")
        recompute_btn = st.button("Recompute attention")

    # Compute attention and cache results in session_state so changing `piece` doesn't rerun model
    if compute_btn or recompute_btn:
        # Load model+encoder only when necessary. Cache in session_state to avoid
        # re-loading on unrelated widget clicks (Prev/Next, sliders, etc.).
        if recompute_btn or ("model" not in st.session_state):
            st.info("Loading model and encoder (cached)")
            model, encoder = load_model_and_encoder(
                model_name, special_name, device=device
            )
            st.session_state["model"] = model
            st.session_state["encoder"] = encoder
        else:
            model = st.session_state.get("model")
            encoder = st.session_state.get("encoder")

        st.info(f"Loading game {game_index} from {pgn}")
        game = vk.load_game_at_index(Path(pgn), int(game_index))

        seq = vk.encode_game(game, encoder, max_tokens=int(max_tokens))
        # build move labels
        board = game.board()
        move_labels = ["<start>"]
        moves = list(game.mainline_moves())
        for move in moves:
            try:
                san = board.san(move)
            except Exception:
                san = move.uci()
            move_labels.append(san)
            board.push(move)
        move_labels.append("<end>")
        if len(move_labels) < len(seq):
            move_labels += ["<pad>"] * (len(seq) - len(move_labels))
        else:
            move_labels = move_labels[: len(seq)]

        tokens = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
        st.info("Computing attention maps (this runs a model forward pass)")
        _, attn_maps = vk.compute_attention_maps(model, tokens)

        # cache in session_state
        st.session_state["attn_maps"] = attn_maps
        st.session_state["seq"] = seq
        st.session_state["move_labels"] = move_labels
        st.session_state["moves"] = moves
        st.session_state["pgn"] = pgn
        st.session_state["game_index"] = int(game_index)
        st.session_state["model_name"] = model_name
        st.session_state["special_name"] = special_name
        st.session_state["max_tokens"] = int(max_tokens)

        layer_ids = [int(x) for x in layers.split(",") if x]
        mats = []
        for layer in layer_ids:
            m = attn_maps[layer][int(head)]
            if isinstance(m, torch.Tensor):
                m = m.detach().cpu().numpy()
            mats.append(np.asarray(m))

        # compute plies (cheap; does not rerun model)
        try:
            plies = vk.track_piece_plies(game, piece)
        except Exception as e:
            st.error(str(e))
            plies = []
        if not plies:
            st.warning(
                f"Piece on {piece} never moved in this game — plotting without markers."
            )

        token_positions = [p + 1 for p in plies if p + 1 < len(seq)]

        fig = plot_per_layer_square(mats, move_labels, token_positions, layer_ids, head)
        st.pyplot(fig)

        # allow download
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200)
        buf.seek(0)
        st.download_button(
            "Download PNG",
            data=buf,
            file_name=f"attn_game{game_index}_piece{piece}_h{head}.png",
        )

        pass

    # Plot using cached attention maps without recomputing the model
    if plot_btn:
        if "attn_maps" not in st.session_state:
            st.error("No cached attention found. Click 'Compute attention' first.")
        else:
            # reload game (cheap) to compute piece plies only
            game = vk.load_game_at_index(Path(pgn), int(game_index))
            try:
                plies = vk.track_piece_plies(game, piece)
            except Exception as e:
                st.error(str(e))
                plies = []
            if not plies:
                st.warning(
                    f"Piece on {piece} never moved in this game — plotting without markers."
                )

            seq = st.session_state["seq"]
            move_labels = st.session_state["move_labels"]
            attn_maps = st.session_state["attn_maps"]
            layer_ids = [int(x) for x in layers.split(",") if x]
            mats = []
            for layer in layer_ids:
                m = attn_maps[layer][int(head)]
                if isinstance(m, torch.Tensor):
                    m = m.detach().cpu().numpy()
                mats.append(np.asarray(m))

            token_positions = [p + 1 for p in plies if p + 1 < len(seq)]
            fig = plot_per_layer_square(
                mats, move_labels, token_positions, layer_ids, head
            )
            st.pyplot(fig)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=200)
            buf.seek(0)
            st.download_button(
                "Download PNG",
                data=buf,
                file_name=f"attn_game{game_index}_piece{piece}_h{head}.png",
            )

            pass

    # Persistent board viewer: always render when we have cached moves so
    # Prev/Next/Slider widgets exist on every rerun (prevents disappearing image).
    if "moves" in st.session_state:
        st.markdown("**Board viewer**")
        moves_list = st.session_state["moves"]
        n_plies = len(moves_list)
        if "viewer_ply" not in st.session_state:
            st.session_state["viewer_ply"] = 0

        col1, col2, col3 = st.columns([1, 4, 1])
        with col1:
            if st.button("Prev", key="viewer_prev"):
                st.session_state["viewer_ply"] = max(
                    0, st.session_state["viewer_ply"] - 1
                )
        with col3:
            if st.button("Next", key="viewer_next"):
                st.session_state["viewer_ply"] = min(
                    n_plies, st.session_state["viewer_ply"] + 1
                )

        # let the slider manage its own state (use key only)
        st.slider("Ply (0 = start)", min_value=0, max_value=n_plies, key="viewer_ply")
        ply = st.session_state["viewer_ply"]

        # render board at given ply by replaying moves (cheap)
        board = chess.Board()
        for i in range(ply):
            board.push(moves_list[i])

        lastmove = moves_list[ply - 1] if ply > 0 else None
        svg = chess.svg.board(
            board=board, size=360, lastmove=lastmove, coordinates=True
        )
        # Show board and PGN side-by-side so the PGN is always visible
        col_board, col_pgn = st.columns([1, 1])
        with col_board:
            if _HAVE_CAIROSVG:
                png = cairosvg.svg2png(bytestring=svg.encode("utf-8"))
                st.image(png)
            else:
                components.html(svg, height=380)

        with col_pgn:
            try:
                game_obj = vk.load_game_at_index(
                    Path(st.session_state.get("pgn")),
                    int(st.session_state.get("game_index")),
                )
                exporter = chess.pgn.StringExporter(
                    headers=True, variations=False, comments=False
                )
                pgn_text = game_obj.accept(exporter)
            except Exception:
                # Fallback: render SAN move list from cached move labels
                move_labels = st.session_state.get("move_labels", [])
                # skip <start> and <end> tokens for compactness
                moves_only = [
                    m for m in move_labels if m not in ("<start>", "<end>", "<pad>")
                ]
                pgn_text = " ".join(moves_only)
            st.text_area("Game PGN", value=pgn_text, height=380)

        # Also render cached attention plot (so it stays visible while interacting)
        # Avoid duplicate rendering when the user just clicked Compute or Plot.
        if ("attn_maps" in st.session_state) and not (compute_btn or plot_btn):
            try:
                attn_maps = st.session_state["attn_maps"]
                seq = st.session_state.get("seq")
                move_labels = st.session_state.get("move_labels")
                layer_ids = [int(x) for x in layers.split(",") if x]
                head_sel = int(head)

                mats = []
                for layer in layer_ids:
                    m = attn_maps[layer][head_sel]
                    if isinstance(m, torch.Tensor):
                        m = m.detach().cpu().numpy()
                    mats.append(np.asarray(m))

                # recompute plies from cached moves/game info (cheap)
                try:
                    game_cached = vk.load_game_at_index(
                        Path(st.session_state.get("pgn")),
                        int(st.session_state.get("game_index")),
                    )
                    plies = vk.track_piece_plies(game_cached, piece)
                except Exception:
                    plies = []
                token_positions = (
                    [p + 1 for p in plies if p + 1 < len(seq)]
                    if seq is not None
                    else []
                )

                fig = plot_per_layer_square(
                    mats, move_labels, token_positions, layer_ids, head_sel
                )
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not render cached attention plot: {e}")


if __name__ == "__main__":
    main()
