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
    train_split = 0.7
    dataset_size = y.shape[0]
    train_size = math.ceil(dataset_size * train_split)
    
    y_train = y[:train_size]
    x_train = x[:train_size]
    y_test = y[train_size:]
    x_test = x[train_size:]
    
    
    nn = build_accelerated_network(
        input_layer_dim=x.shape[1],
        hidden_layer_shapes=(128, 64, 32, 1), 
        activation=ActivationFunc.sigmoid,
        seed=random.randint(0,512), 
        runtime=AcceleratedRuntime.numba
    )
    
    config = AcceleratedTrainingConfig(
        learning_rate=0.1,
        max_power=17,
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

    final_step = result.milestone_steps[-1]
    test_scores = result.snapshots[final_step].reshape(-1)
    test_targets = result.evaluation_targets.reshape(-1)
    test_predictions = (test_scores >= 0.5).astype(test_targets.dtype)
    test_accuracy = np.mean(test_predictions == test_targets)

    print(f"Reserved test loss: {result.losses[final_step]:.6f}")
    print(f"Reserved test accuracy: {test_accuracy:.4%}")



if __name__ == "__main__":
    main()
