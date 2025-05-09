import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ConfigRecord
from fl_brain.task import Net, get_weights, load_data, set_weights, test, train


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, strategy_name, context: Context):
        self.client_state = context.state
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.strategy_name = strategy_name
        self.proximal_mu = context.run_config["proximal_mu"]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        
        if "fit_metrics" not in self.client_state.config_records:
            self.client_state.config_records["fit_metrics"] = ConfigRecord()

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        torch.cuda.empty_cache()
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            config["lr"],
            self.device,
            self.strategy_name,
            self.proximal_mu,
        )

        # Append to persistent state the `train_loss` just obtained
        fit_metrics = self.client_state.config_records["fit_metrics"]
        if "train_loss_hist" not in fit_metrics:
            # If first entry, create the list
            fit_metrics["train_loss_hist"] = [train_loss]
        else:
            # If it's not the first entry, append to the existing list
            fit_metrics["train_loss_hist"].append(train_loss)

        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss}
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    # Load model and data
    data_dir = "/media/sslab/PACS/sslab/nguyentiendung/data_processed/participants.xlsx"
    net = Net()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    partitioner = context.run_config["partitioner"]
    trainloader, valloader = load_data(partition_id, num_partitions, partitioner, data_dir)
    local_epochs = context.run_config["local-epochs"]
    strategy_name = context.run_config["strategy"]

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs, strategy_name, context).to_client()


# Create ClientApp
app = ClientApp(client_fn=client_fn, )
