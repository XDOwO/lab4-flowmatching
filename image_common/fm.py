from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def expand_t(t, x):
    """Helper function to expand t to the same number of dimensions as x."""
    for _ in range(x.ndim - 1):
        t = t.unsqueeze(-1)
    return t


class FMScheduler(nn.Module):
    def __init__(self, num_train_timesteps=1000, sigma_min=0.001):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.sigma_min = sigma_min

    def uniform_sample_t(self, batch_size) -> torch.LongTensor:
        """Samples a random timestep t from [0, 1] for training."""
        # We divide by num_train_timesteps to normalize t to [0, 1)
        ts = (
            np.random.choice(np.arange(self.num_train_timesteps), batch_size)
            / self.num_train_timesteps
        )
        return torch.from_numpy(ts)

    def compute_psi_t(self, x1, t, x):
        """
        Compute the conditional flow psi_t(x | x_1).

        Note that time flows in the opposite direction compared to DDPM/DDIM.
        As t moves from 0 to 1, the probability paths shift from a prior distribution p_0(x)
        to a more complex data distribution p_1(x).

        Input:
            x1 (`torch.Tensor`): Data sample from the data distribution (the target image).
            t (`torch.Tensor`): Timestep in [0,1).
            x (`torch.Tensor`): The input from the prior distribution (the noise, x0).
        Output:
            psi_t (`torch.Tensor`): The conditional flow at t.
        """
        # t is expanded to match the dimensions of x1 (e.g., [B, C, H, W])
        t = expand_t(t, x1)

        ######## TODO ########
        # DO NOT change the code outside this part.
        # Compute psi_t(x) using the "Rectified Flow" interpolation path.
        # 
        # Hint: The interpolation formula is x_t = (1-t) * x_0 + t * x_1
        # where 'x' is x_0 (noise) and 'x1' is the data sample.

        psi_t = (1 - t) * x + t * x1
        ######################

        return psi_t

    def step(self, xt, vt, dt):
        """
        The simplest ode solver as the first-order Euler method:
        x_next = xt + dt * vt
        """

        ######## TODO ########
        # DO NOT change the code outside this part.
        # Implement each step of the first-order Euler method.
        #
        # Hint: Use the formula x_next = xt + dt * vt
        # where:
        #   xt: current position (e.g., x_t)
        #   vt: predicted velocity (the direction to move)
        #   dt: time step size
        
        x_next = xt + dt * vt  # Implement the Euler step
        ######################

        return x_next


class FlowMatching(nn.Module):
    """
    This class represents the multi-step Flow Matching model (the "teacher").
    It is trained using the CFM objective and uses an ODE solver (Euler)
    and Classifier-Free Guidance (CFG) for sampling.
    """
    def __init__(self, network: nn.Module, fm_scheduler: FMScheduler, **kwargs):
        super().__init__()
        self.network = network
        self.fm_scheduler = fm_scheduler

    @property
    def device(self):
        return next(self.network.parameters()).device

    @property
    def image_resolution(self):
        return self.network.image_resolution

    def get_loss(self, x1, class_label=None, x0=None):
        """
        The conditional flow matching objective, corresponding Eq. 23 in the FM paper.
        This is the training for the "teacher" model.
        """
        batch_size = x1.shape[0]
        # Sample a random time t in [0, 1)
        t = self.fm_scheduler.uniform_sample_t(batch_size).to(x1)
        if x0 is None:
            x0 = torch.randn_like(x1) # Sample noise from prior

        ######## TODO ########
        # DO NOT change the code outside this part.
        # Implement the CFM (Conditional Flow Matching) objective.
        #
        # Steps to complete:
        # 1. Reshape t for broadcasting (from [B] to [B, 1, 1, 1])
        #    Hint: use t.view(-1, *([1] * (x1.dim() - 1)))
        #
        # 2. Create the interpolated sample x_t = (1-t)*x_0 + t*x_1
        #    This is a point on the straight-line path between noise and image.
        #
        # 3. Define the target velocity u_t = x_1 - x_0
        #    This is the constant velocity vector for the straight-line path.
        #
        # 4. Get the model's predicted velocity v(x_t, t)
        #    Call self.network(x_t, t) or self.network(x_t, t, class_label=class_label)
        #
        # 5. Calculate the loss: MSE between predicted and target velocity
        #    Hint: use .pow(2).mean()

        xt = self.fm_scheduler.compute_psi_t(x1, t, x0)

        if class_label is not None:
            model_out = self.network(xt, t, class_label=class_label)
        else:
            model_out = self.network(xt, t)

        target = x1 - x0

        loss = F.mse_loss(model_out, target)
        ######################

        return loss

    def conditional_psi_sample(self, x1, t, x0=None):
        """Helper to sample a point on the flow path."""
        if x0 is None:
            x0 = torch.randn_like(x1)
        return self.fm_scheduler.compute_psi_t(x1, t, x0)

    @torch.no_grad()
    def sample(
        self,
        shape,
        num_inference_timesteps=50,
        return_traj=False,
        class_label: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = 1.0,
        verbose=False,
    ):
        """
        Inference sampling loop for the multi-step "teacher" model.
        """
        batch_size = shape[0]
        # Start from pure noise (x_0 at t=0)
        x_T = torch.randn(shape).to(self.device)
        do_classifier_free_guidance = guidance_scale > 1.0

        if do_classifier_free_guidance:
            assert class_label is not None
            assert (
                len(class_label) == batch_size
            ), f"len(class_label) != batch_size. {len(class_label)} != {batch_size}"

        traj = [x_T]

        timesteps = [
            i / num_inference_timesteps for i in range(num_inference_timesteps)
        ]
        # Create a list of time tensors, e.g., [t=0.0, t=0.02, t=0.04, ...]
        timesteps = [torch.tensor([t] * x_T.shape[0]).to(x_T) for t in timesteps]
        pbar = tqdm(timesteps) if verbose else timesteps
        xt = x_T # xt is our current position, starting at x_0
        for i, t in enumerate(pbar):
            # Get the next time step, or t=1.0 if we're at the end
            t_next = timesteps[i + 1] if i < len(timesteps) - 1 else torch.ones_like(t)
            

            ######## TODO ########
            # Complete the sampling loop
            #
            # Steps to complete:
            # 1. Calculate the time step size (dt = t_next - t)
            #    Then expand dt using expand_t(dt, xt) to match xt's dimensions
            #
            # 2. Predict the velocity (vt) at the current time t
            #    - If using CFG (do_classifier_free_guidance is True):
            #      a) Get conditional prediction: v_cond = self.network(xt, t, class_label=class_label)
            #      b) Get unconditional prediction: v_uncond = self.network(xt, t, class_label=None)
            #      c) Combine using CFG formula: vt = v_uncond + guidance_scale * (v_cond - v_uncond)
            #    - Otherwise:
            #      d) Just get standard prediction: vt = self.network(xt, t, class_label=class_label)
            #
            # 3. Perform one Euler step to get x_next
            #    Use: xt_next = self.fm_scheduler.step(xt, vt, dt)
            #
            # 4. Update xt for the next iteration: xt = xt_next

            if do_classifier_free_guidance:
                uncond = self.network(xt, t)
                cond = self.network(xt, t, class_label=class_label)
                v = uncond + guidance_scale * (cond - uncond)
            else:
                if class_label is not None:
                    v = self.network(xt, t, class_label=class_label)
                else:
                    v = self.network(xt, t)

            dt = t_next - t
            dt = expand_t(dt, xt)

            xt = self.fm_scheduler.step(xt, v, dt)
            ######################

            traj[-1] = traj[-1].cpu()
            traj.append(xt.clone().detach())
        if return_traj:
            return traj
        else:
            return traj[-1]

    def save(self, file_path):
        hparams = {
            "network": self.network,
            "fm_scheduler": self.fm_scheduler,
        }
        state_dict = self.state_dict()

        dic = {"hparams": hparams, "state_dict": state_dict}
        torch.save(dic, file_path)

    def load(self, file_path):
        dic = torch.load(file_path, map_location="cpu")
        hparams = dic["hparams"]
        state_dict = dic["state_dict"]

        self.network = hparams["network"]
        self.fm_scheduler = hparams["fm_scheduler"]

        self.load_state_dict(state_dict)
