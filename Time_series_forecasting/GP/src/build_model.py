#! /usr/bin/env python3
import numpy as np
import gpytorch
import torch
from matplotlib import pyplot as plt
from gpytorch.kernels import MaternKernel, LinearKernel, ScaleKernel
torch.set_default_dtype(torch.float64)


####################################################################
#                      Multitask Exact GP                          #
####################################################################

# define a GP model using exact inference with constant mean function and squared exponential (RBF) kerclass ExactGPModel(gpytorch.models.ExactGP):
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self,
                 train_x,
                 train_y,
                 likelihood,
                 mean = gpytorch.means.LinearMean(input_size=1),
                 lengthscale_prior = None,
                 lengthscale_minconstraint = None,
                 lengthscale_mult: float = 1.0):
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = mean

        if isinstance(self.mean_module, gpytorch.means.LinearMean):
            with torch.no_grad():
                X = train_x if train_x.ndim == 2 else train_x.unsqueeze(-1)  # [N,D]
                X1 = torch.cat([torch.ones(X.size(0), 1, dtype=X.dtype), X], dim=1)  # [N,D+1]
                beta = torch.linalg.lstsq(X1, train_y.reshape(-1,1)).solution.squeeze(1)  # [D+1]
                a, w = beta[0], beta[1:]   # 截距、权重
                self.mean_module.bias.copy_(a.view_as(self.mean_module.bias))
                self.mean_module.weights.copy_(w.view_as(self.mean_module.weights))


        # ---- 可选：lengthscale 下界约束 ----
        lengthscale_constraint = None
        if lengthscale_minconstraint is not None:
            xvals = torch.unique(train_x.reshape(-1)).sort().values
            if xvals.numel() >= 2:
                gaps = torch.diff(xvals)
                stat_map = {'min': torch.min, 'mean': torch.mean, 'median': torch.median, 'max': torch.max}
                if lengthscale_minconstraint not in stat_map:
                    raise ValueError(f"lengthscale_minconstraint must be one of {list(stat_map.keys()) + [None]}")
                Constt = float(lengthscale_mult) * float(stat_map[lengthscale_minconstraint](gaps).item())
                lengthscale_constraint = gpytorch.constraints.GreaterThan(Constt)

        # ---- 核：Matern(ν=1.5) + 线性核 ----
        ard_dims = train_x.shape[-1] if train_x.ndim == 2 else 1

        k_time = MaternKernel(
            nu=1.5,
            ard_num_dims=ard_dims,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=lengthscale_constraint,
        )

        # 如果 X 只有“时间”一列，下面这行直接用就行；
        # 如果 X 是多维（时间+协变量），且只想让“时间”带线性趋势：
        # k_lin = LinearKernel(active_dims=(0,))   # 假设第0列是时间
        k_lin = LinearKernel()

        self.covar_module = ScaleKernel(k_time + k_lin)

        # ---- 初始化（外推更稳）----
        with torch.no_grad():
            init_ls = 0.3
            if ard_dims == 1:
                k_time.lengthscale = torch.tensor(init_ls, dtype=train_x.dtype)
            else:
                k_time.lengthscale = torch.full((ard_dims,), init_ls, dtype=train_x.dtype)

            self.covar_module.outputscale = torch.tensor(1.0, dtype=train_x.dtype)
            likelihood.noise = torch.tensor(0.05 ** 2, dtype=train_x.dtype)

            # 线性核的强度给个起点
            k_lin.variance = torch.tensor(1.0, dtype=train_x.dtype)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
# Model definition
def building_exactgp_model(tpptr_df, proteins2test, conds, lengthscale_prior, lengthscale_minconstraint, lengthscale_mult, mean = gpytorch.means.LinearMean(input_size=1)):
    """
    Defines a GP model using exact inference with zero mean function and squared exponential (RBF) kernel class ExactGPModel(gpytorch.models.ExactGP)
    Args:
        - tpptr_df (pandas.core.frame.DataFrame):
            Expected is a pandas data frame with variables
            'uniqueID' for protein ids,
            'x' for profiled temperatures,
            'y' for scaled protein intensities,
            'condition' for performed comparison (e.g. vehicle vs drug)
            
            Note that the input data needs to be already filtered and formatted (e.g. proteins that should be analyzed need to be identified in both conditions).
.
        - conds (numpy.ndarray): array containing conditions that melting curves will be modeled for.
        - lengthscale_prior (str): Set this if you want to apply a prior to the lengthscale parameter. (Default: None). See: https://docs.gpytorch.ai/en/stable/priors.html for possible priors.
        - lengthscale_minconstraint (str): Set this if you want to apply a constraint to the lengthscale parameter. Possible options are: 'min', 'mean', 'median', 'max', None.
            
            This constraint will limit the lowest possible value for the lengthscale (implemented via gpytorch.constraints.GreaterThan) and was adopted from:
                LeSueur, Cecile, Magnus Rattray, and Mikhail Savitski. 2023. “Hierarchical Gaussian Process Models Explore the Dark Meltome of Thermal Proteome Profiling Experiments.” bioRxiv. https://doi.org/10.1101/2023.10.26.564129.
        
        - mean (mean function): Defines the mean function of the GP. (Default: gpytorch.means.ZeroMean()). See: https://docs.gpytorch.ai/en/stable/means.html for alternatives.
            
    Returns:
        gpytorch.models.model_list.IndependentModelList: ModelList of independent GP models. 
    For more details, see the GPyTorch documentation:
    https://docs.gpytorch.ai/en/stable/examples/03_Multitask_Exact_GPs/ModelList_GP_Regression.html
    """
    model_list = []
    l_list = []
    metadata_list = []  # List to store metadata
    for prot in proteins2test:
        # filter data
        df = tpptr_df[tpptr_df['uniqueID'] == prot]
        # define alternative models (one model for each condition)
        for cond in df['condition'].unique():
            dfcond = df[df['condition'] == cond]
            temp = torch.as_tensor(np.asarray(dfcond['x'])).double()
            intens = torch.as_tensor(np.asarray(dfcond['y'])).double()
            lik = gpytorch.likelihoods.GaussianLikelihood(
                noise_prior=gpytorch.priors.GammaPrior(2.0, 200.0),           # 均值≈0.01（logit 空间）
                noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
)
            l_list.append(lik)
            model = ExactGPModel(temp, intens, lik, mean, lengthscale_prior, lengthscale_minconstraint, lengthscale_mult)
            model_list.append(model)
            model_id = len(model_list)
            metadata_list.append((prot, cond, model_id))
    # dimensions
    n_models = len(model_list)
    n_cond = len(conds)
    n_prot = len(proteins2test)
    assert n_models == n_cond * n_prot, "Error in building the model list."
    
    # create ModelList Multioutput GP
    return model_list, l_list, metadata_list, n_cond, n_prot, n_models 
