from scipy.io.arff import test
from ffnnpy.neural_net.accelerated import fit_dataset_accelerated
import read_htru2_arff
import numpy as np
import math
import random


from ffnnpy.neural_net import (
    AcceleratedRuntime,
    AcceleratedTrainingConfig,
    ActivationFunc,
    AsyncProgressPrinter,
    build_accelerated_network,
)



def main():
    x, y, labels = read_htru2_arff.load_htru2()
    
    #split inputs 80% train/20% test
    train_split = 0.8
    dataset_size = y.shape[0]
    train_size = math.ceil(dataset_size * train_split)
    
    y_train = y[:train_size]
    x_train = x[:train_size]
    y_test = y[train_size:]
    x_test = x[train_size:]
    
    
    nn = build_accelerated_network(
        input_layer_dim=x.shape[1],
        hidden_layer_shapes=(128, 128, 64, 8, 1), 
        activation=ActivationFunc.sigmoid,
        seed=np.random.random_integers(low=0, high=512, size=1)[0], 
        runtime=AcceleratedRuntime.numba
    )
    
    config = AcceleratedTrainingConfig(
        learning_rate=0.01,
        max_power=16,
        evaluation_points=512,
        runtime=AcceleratedRuntime.numba
    )

    with AsyncProgressPrinter(enabled=True) as logger:
        result = fit_dataset_accelerated(
            network=nn,
            train_inputs=x_train,
            train_targets=y_train,
            config=config,
            evaluation_inputs=x_test,
            evaluation_targets=y_test,
            progress_logger=logger.log
        )


if __name__ == "__main__":
    main()
