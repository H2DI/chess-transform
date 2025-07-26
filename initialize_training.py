from configs import TrainingSession, TrainingConfig, ModelConfig
from chess_seq.training.trainer_runner import ChessTrainerRunner


model_config = ModelConfig()
training_config = TrainingConfig()
training_session = TrainingSession()

assert training_session.new_model

trainer_runner = ChessTrainerRunner(
    training_session, model_config=model_config, training_config=training_config
)

num_params = sum(p.numel() for p in trainer_runner.model.parameters())
print(f"Number of parameters in the model: {num_params}")
