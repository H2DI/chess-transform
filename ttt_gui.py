import tkinter as tk
from tkinter import messagebox

import chess_seq.utils as utils
from chess_seq.tictactoe.mechanics import TTTBoard
from chess_seq.tictactoe.players import NNPlayer


class TTTGui:
    def __init__(self, player, agent_first=False, greedy=True):
        self.player = player
        self.greedy = greedy
        self.board = TTTBoard()
        self.root = tk.Tk()
        self.root.title("Tic-Tac-Toe")
        self.buttons = []
        self._build_board()
        self._update_board()
        self.agent_first = agent_first
        self.human_turn = not agent_first  # Human starts as X unless agent_first

        # Add Restart button
        restart_btn = tk.Button(self.root, text="Restart", command=self._restart_game)
        restart_btn.grid(row=3, column=0, columnspan=3, sticky="nsew")

        if self.agent_first:
            self.root.after(100, self._agent_move)

    def _build_board(self):
        for i in range(3):
            row = []
            for j in range(3):
                btn = tk.Button(
                    self.root,
                    text="",
                    width=6,
                    height=3,
                    command=lambda x=i, y=j: self._on_click(x, y),
                )
                btn.grid(row=i, column=j)
                row.append(btn)
            self.buttons.append(row)

    def _on_click(self, i, j):
        if not self.human_turn or self.board.is_game_over():
            return
        move = (i, j)
        if move not in self.board.legal_moves:
            return
        self.board.push(move)
        self._update_board()
        self.human_turn = False
        if self.board.is_game_over():
            self._show_winner()
        else:
            self.root.after(500, self._agent_move)

    def _agent_move(self):
        move = self.player.get_move(self.board, greedy=self.greedy)
        print(f"Agent plays: {move}")
        if move in self.board.legal_moves:
            self.board.push(move)
        self._update_board()
        self.human_turn = True
        if self.board.is_game_over():
            self._show_winner()

    def _update_board(self):
        for i in range(3):
            for j in range(3):
                val = self.board.X[i, j] + 2 * self.board.O[i, j]
                self.buttons[i][j].config(text={0: "", 1: "X", 2: "O"}.get(val, ""))

    def _show_winner(self):
        winner = self.board.winner
        msg = "Draw!" if winner == "T" else f"Winner: {winner}"
        tk.messagebox.showinfo("Game Over", msg)

    def _restart_game(self):
        self.board = TTTBoard()
        self._update_board()
        self.human_turn = not self.agent_first
        if self.agent_first:
            self.root.after(100, self._agent_move)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    # If these lines block or crash, you'll now get a dialog/traceback
    model, encoder, checkpoint = utils.load_model(
        "ttt_large_573440_GRPO", special_name="no_loss"
    )
    player = NNPlayer(model, encoder, mask_illegal=True)

    gui = TTTGui(player, agent_first=True, greedy=True)  # Agent plays first
    gui.run()
