import torch
from typing import List, Tuple
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from torch.utils.data import DataLoader

from fl_brain.task import Net, get_weights, set_weights, test
from fl_brain.my_strategy import CustomFedAvg, CustomFedProx, CustomFedAdam


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregates metrics from an evaluate round."""

    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)

    return {"accuracy": sum(accuracies) / total_examples}


def on_fit_config(server_round: int) -> Metrics:
    """Adjusts learning rate based on current round."""

    lr_initial = 0.01
    lr_decay = 0.93  # Decay rate
    lr = lr_initial * (lr_decay ** (server_round - 1))

    return {"lr": lr}


def server_fn(context: Context):
    """A function that creates the components for a ServerApp."""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Read training config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    strategy_name = context.run_config["strategy"]
    partitioner = context.run_config["partitioner"]

    # Initial model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Strategy selection
    if strategy_name == "fedavg":
        strategy = CustomFedAvg(
            num_rounds=num_rounds,
            partitioner=partitioner,
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            min_available_clients=2,
            initial_parameters=parameters,
            evaluate_metrics_aggregation_fn=weighted_average,
            on_fit_config_fn=on_fit_config,
        )
    elif strategy_name == "fedprox":
        strategy = CustomFedProx(
            num_rounds=num_rounds,
            partitioner=partitioner,
            proximal_mu=0.1,
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            min_available_clients=2,
            initial_parameters=parameters,
            evaluate_metrics_aggregation_fn=weighted_average,
            on_fit_config_fn=on_fit_config,
        )
    elif strategy_name == "fedadam":
        strategy = CustomFedAdam(
            num_rounds=num_rounds,
            partitioner=partitioner,
            eta=0.1,       # Server learning rate (0.01-0.1)
            eta_l=0.01,    # Client learning rate (<eta)
            beta_1=0.9,    # Momentum
            beta_2=0.99,   # Second moment
            tau=1e-9,      # Stability term
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            min_available_clients=2,
            initial_parameters=parameters,
            evaluate_metrics_aggregation_fn=weighted_average,
            on_fit_config_fn=on_fit_config,
        )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)