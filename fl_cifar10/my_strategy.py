from flwr.common import FitRes, Parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg, FedProx, FedAdam
from fl_torch.task import Net, set_weights

import torch
import json


class BaseCustomStrategy:
    def __init__(self, strategy_name: str, partitioner: str, num_rounds: int, json_path: str):
        self.strategy_name = strategy_name
        self.partitioner = partitioner
        self.num_rounds = num_rounds
        self.json_path = json_path
        self.results_to_save = {}

    def save_model_if_final(self, parameters: Parameters, server_round: int, filename: str):
        if server_round == self.num_rounds:
            ndarrays = parameters_to_ndarrays(parameters)
            model = Net()
            set_weights(model, ndarrays)
            torch.save(model.state_dict(), filename)

    def save_and_log_metrics(self, server_round: int, loss: float, metrics: dict):
        result = {"loss": loss, **metrics}
        self.results_to_save[server_round] = result
        with open(self.json_path, "w") as f:
            json.dump(self.results_to_save, f, indent=4)


class CustomFedAvg(FedAvg):
    """FedAvg strategy with model saving and metrics logging."""

    def __init__(self, num_rounds, partitioner, *args, **kwargs):
        super().__init__(*args, **kwargs)
        json_path = f"results_fedavg_{partitioner}.json"
        self.model_filename = f"global_model_final_fedavg_{partitioner}.pt"
        self.helper = BaseCustomStrategy("fedavg", partitioner, num_rounds, json_path)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, bool | bytes | float | int | str]]:
        """Aggregate model updates and save global model checkpoint."""

        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )
        self.helper.save_model_if_final(   
            parameters_aggregated, server_round, self.model_filename
        )

        return parameters_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> tuple[float, dict[str, bool | bytes | float | int | str]] | None:
        """Evaluate global model and save metrics to a local JSON file."""

        loss, metrics = super().evaluate(server_round, parameters)
        self.helper.save_and_log_metrics(server_round, loss, metrics)

        return loss, metrics
    

class CustomFedProx(FedProx):
    """FedProx strategy with model saving and metrics logging."""
    
    def __init__(self, num_rounds, partitioner, proximal_mu, *args, **kwargs):
        super().__init__(proximal_mu=proximal_mu, *args, **kwargs)
        json_path = f"results_fedprox_{partitioner}.json"
        self.model_filename = f"global_model_final_fedprox_{partitioner}.pt"
        self.helper = BaseCustomStrategy("FedProx", partitioner, num_rounds, json_path)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, bool | bytes | float | int | str]]:
        """Aggregate model updates with proximal term and save final model."""

        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )
        self.helper.save_model_if_final(
            parameters_aggregated, server_round, self.model_filename
        )

        return parameters_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> tuple[float, dict[str, bool | bytes | float | int | str]] | None:
        """Evaluate global model and save metrics to a local JSON file."""
        
        loss, metrics = super().evaluate(server_round, parameters)
        self.helper.save_and_log_metrics(server_round, loss, metrics)

        return loss, metrics
    

class CustomFedAdam(FedAdam):
    """FedAdam strategy with model saving and metrics logging."""
    
    def __init__(self, num_rounds, partitioner, eta, eta_l, beta_1, beta_2, tau, *args, **kwargs):
        super().__init__(
            eta=eta,          # Server-side learning rate
            eta_l=eta_l,      # Client-side learning rate
            beta_1=beta_1,    # Momentum parameter
            beta_2=beta_2,    # Second moment parameter
            tau=tau,          # Stability term
            *args,
            **kwargs
        )
        json_path = f"results_fedadam_{partitioner}.json"
        self.model_filename = f"global_model_final_fedadam_{partitioner}.pt"
        self.helper = BaseCustomStrategy("fedadam", partitioner, num_rounds, json_path)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, bool | bytes | float | int | str]]:
        """Aggregate updates with FedAdam's adaptive optimization and save model."""

        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )
        self.helper.save_model_if_final(
            parameters_aggregated, server_round, self.model_filename
        )

        return parameters_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> tuple[float, dict[str, bool | bytes | float | int | str]] | None:
        """Evaluate global model and save metrics to a local JSON file."""

        loss, metrics = super().evaluate(server_round, parameters)
        self.helper.save_and_log_metrics(server_round, loss, metrics)

        return loss, metrics