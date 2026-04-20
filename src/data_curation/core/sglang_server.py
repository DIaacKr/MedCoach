import logging
from sglang.utils import launch_server_cmd, terminate_process, wait_for_server

class SGLangServer:
    def __init__(self, model_path, port=30000, tp_size=1, dp_size=1, log_level="info", seed=42, mem_fraction_static=0.9):
        self.model_path = model_path
        self.port = port
        self.tp_size = tp_size
        self.log_level = log_level
        self.seed = seed
        self.mem_fraction_static = mem_fraction_static
        self.server_process = None
        self.dp_size = dp_size

    def start(self):
        server_cmd = (
            f"python -m sglang.launch_server "
            f"--model-path {self.model_path} "
            f"--tp {self.tp_size} "
            f"--dp {self.dp_size} "
            f"--log-level {self.log_level} "
            f"--random-seed {self.seed} "
            f"--mem-fraction-static {self.mem_fraction_static}"
        )
        self.server_process, port = launch_server_cmd(server_cmd, port=self.port)
        
        if port != self.port:
            self.terminate()
            raise ValueError(f"Server port mismatch: expected {self.port}, got {port}")

        wait_for_server(f"http://localhost:{port}")
        logging.info(f"Server started at http://localhost:{port}")

    def terminate(self):
        if self.server_process:
            terminate_process(self.server_process)
            logging.info("Server terminated")
