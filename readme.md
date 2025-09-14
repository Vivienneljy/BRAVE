# BRAVE

This is the implmentation of "Breaking the Benign Majority: A Variance-Based Evaluator to Find Malicious Client of Label-Flipping Attacks in Federated Learning". 

## Code Structure

The architecture of this code is as fowllowed:

    FedCAR    
        |_ training
        |    |_ aggregate.py: mechanisms of global aggregation
        |    |_ update.py: local training and inference
        |_ preprocessing
        |    |_ dataloader.py: load local datasets and launch label flipping attacks
        |    |_ datasets.py: data preprocessing
        |    |_ sample.py: distribute the dataset to clients
        |    |_ model: model architecture for MNIST and CIFAR-10
        |_ defending: anomaly detection methods
        |_ util
        |    |_ log_utils.py: generate log files
        |    |_ metrics.py: evaluation metrics
        |    |_ options.py: hyperparameter configuration
        |    |_ param_utils.py: model parameter conversion tool
        |_ main.py: startup file

## Getting Started

1. Set hyperparameters in `util\options.py`
2. Start training with `train.py`
3. View training log: In the terminal, navigate to the specified log folder using `cd YourLogName`, and view the logs in TensorBoard with `tensorboard --logdir=figs`.

## Hyperparameters

- To use BRAVE as the defense, set `defense='brave'` for clustering, and `idea1=True` for identifying malicious clusters
- To integrate BRAVE with other anomaly detection methods (FLAME, XMAM, LFighter,DPFLA), set `idea1=True`, and `defense='flame/xmam/lfighter/dpfla'`
