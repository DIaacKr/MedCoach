import logging
import torch
from typing import List
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

class MemoryCleanupCallback(TrainerCallback):
    def __init__(self, cleanup_steps=50):
        self.cleanup_steps = cleanup_steps
        logging.info(f"MemoryCleanupCallback initialized. Will clear CUDA cache every {cleanup_steps} steps on main process.")

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_world_process_zero and self.cleanup_steps > 0 and state.global_step % self.cleanup_steps == 0:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                logging.error(f"Step {state.global_step} [MemoryCleanupCallback]: Error clearing CUDA cache: {str(e)}")


def get_callbacks(memory_cleanup_steps=50) -> List[TrainerCallback]:
    callbacks = []

    # Memory Cleanup Callback
    if memory_cleanup_steps > 0:
        memory_cleanup_callback = MemoryCleanupCallback(cleanup_steps=memory_cleanup_steps)
        callbacks.append(memory_cleanup_callback)

    logging.info(f"Initialized {len(callbacks)} callbacks.")
    return callbacks
