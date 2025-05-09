import torch
from typing import List, Tuple
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from datasets import load_dataset
from torch.utils.data import DataLoader

from fl_cifar10.task import Net, get_weights, set_weights, test, get_transforms
from fl_cifar10.my_strategy import CustomFedAvg, CustomFedProx, CustomFedAdam


def get_evaluate_fn(testloader, device):
    """Return a callback that evaluates the global model."""

    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model using provided centralised testset."""

        net = Net()
        set_weights(net, parameters_ndarrays)
        net.to(device)
        loss, accuracy = test(net, testloader, device)

        return loss, {"cen_accuracy": accuracy}

    return evaluate


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

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Load global test set
    testset = load_dataset("uoft-cs/cifar10")["test"]
    testloader = DataLoader(testset.with_transform(get_transforms()), batch_size=32, shuffle=False)
    
    # Define and configure the strategy
    strategy = context.run_config["strategy"]
    partitioner = context.run_config["partitioner"]
    if strategy == "fedavg":
        strategy = CustomFedAvg(
            num_rounds=num_rounds,
            partitioner=partitioner,
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            min_available_clients=2,
            initial_parameters=parameters,
            evaluate_metrics_aggregation_fn=weighted_average,
            on_fit_config_fn=on_fit_config,
            evaluate_fn=get_evaluate_fn(testloader, device=device),
        )
    elif strategy == "fedprox":
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
            evaluate_fn=get_evaluate_fn(testloader, device=device),
        )
    elif strategy == "fedadam":
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
            evaluate_fn=get_evaluate_fn(testloader, device=device),
        )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)