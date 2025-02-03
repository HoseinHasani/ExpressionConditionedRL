import numpy as np
import pysindy as ps
from task_inference_utils.base_inference import BaseTaskInference

class SymbolicRegressionInference(BaseTaskInference):
    def __init__(self, context_size, feature_lib=None, optimizer=None, ensemble=True):
        """
        Initialize the Symbolic Regression Inference module.
        
        Args:
            context_size (int): The size of the context vector.
            feature_lib: Pysindy feature library. Defaults to PolynomialLibrary with degree 1.
            optimizer: Pysindy optimizer. Defaults to STLSQ.
            ensemble (bool): Whether to use an ensemble model. Defaults to True.
        """
        super().__init__(context_size)
        self.feature_lib = feature_lib or ps.PolynomialLibrary(degree=1, include_bias=True)
        self.optimizer = optimizer or ps.STLSQ(threshold=0.02, alpha=0.15, verbose=False, max_iter=40)
        self.ensemble = ensemble


    def infer_task(self, trajectory_buffer):
        if len(trajectory_buffer) < 5:
            return np.zeros(self.context_size, dtype=np.float32)

        states, actions, rewards = zip(*trajectory_buffer)
        states = np.array(states)
        actions = np.array(actions)

        x_train = states[:-1]
        u_train = actions[:-1]
        y_train = states[1:][:, :2] 

        model = ps.SINDy(discrete_time=True,
                         feature_library=self.feature_lib,
                         optimizer=self.optimizer)  

        if self.ensemble:
            model.fit(x_train, u=u_train, x_dot=y_train, ensemble=self.ensemble,
                  multiple_trajectories=False, n_models=10, quiet=True)
        else:
            model.fit(x_train, u=u_train, x_dot=y_train, ensemble=self.ensemble,
                  multiple_trajectories=False, quiet=True)

        coeffs = model.coefficients()
        flat_coeffs = np.concatenate(coeffs, axis=None).astype(np.float32)
        
        # flat_coeffs = np.clip(flat_coeffs, -1.1, 1.1)
        flat_coeffs = 2 * np.tanh(0.5 * flat_coeffs)

        return flat_coeffs