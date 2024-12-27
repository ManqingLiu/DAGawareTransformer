import torch.nn as nn

def g_formula_loss_fun(outputs, labels, return_items=True):
    loss = []
    batch_items = {}

    # Continuous values - use MSE loss for MLE
    batch_loss = nn.functional.mse_loss(outputs, labels)

    if return_items:
        batch_items['y'] = batch_loss.item()
    loss.append(batch_loss)

    loss = sum(loss) / len(loss)
    if return_items:
        return loss, batch_items
    else:
        return loss


def ipw_loss_fun(outputs, labels, return_items=True):
    loss = []
    batch_items = {}

    # For treatment/propensity score estimation, use binary cross entropy
    batch_loss = nn.functional.binary_cross_entropy(
        outputs, labels)

    if return_items:
        batch_items['t'] = batch_loss.item()
    loss.append(batch_loss)

    loss = sum(loss) / len(loss)
    if return_items:
        return loss, batch_items
    else:
        return loss

def aipw_loss_fun(outputs_y, labels_y, outputs_t, labels_t, return_items=True):
    ## average loss of MSE loss for y and BCE loss for t
    loss = []
    batch_items = {}

    # Continuous values - use MSE loss for MLE
    batch_loss_y = nn.functional.mse_loss(outputs_y, labels_y)
    batch_loss_t = nn.functional.binary_cross_entropy_with_logits(
        outputs_t, labels_t)

    if return_items:
        batch_items['y'] = batch_loss_y.item()
        batch_items['t'] = batch_loss_t.item()
    loss.append(batch_loss_y)
    loss.append(batch_loss_t)

    loss = sum(loss) / len(loss)
    if return_items:
        return loss, batch_items


def pdist2sq(X, Y):
    """ Computes the squared Euclidean distance between two matrices """
    C = -2 * torch.mm(X, Y.t())
    nx = torch.sum(X ** 2, dim=1, keepdim=True)
    ny = torch.sum(Y ** 2, dim=1, keepdim=True)
    D = (C + ny.t()) + nx
    return D


def safe_sqrt(x, lbound=1e-5):
    ''' Numerically safe version of PyTorch sqrt '''
    return torch.sqrt(torch.clamp(x, min=lbound))


def ensure_non_empty(X, eps=1e-6):
    """ Ensures that a tensor is non-empty by adding a small epsilon tensor if necessary """
    if X.size(0) == 0:
        X = torch.full((1, X.size(1)), eps, device=X.device)
    return X


def wasserstein(X, t, p, lam=1, its=50, backpropT=False, eps=1e-6):
    """ Returns the Wasserstein distance between treatment groups """
    Xt = X[t == 1]
    Xc = X[t == 0]
    nc = float(Xc.size(0))
    nt = float(Xt.size(0))

    # Add eps to Xt or Xc if they are empty
    Xt = ensure_non_empty(Xt, eps)
    Xc = ensure_non_empty(Xc, eps)

    # Compute distance matrix
    M = safe_sqrt(pdist2sq(Xt, Xc), eps)

    # Estimate lambda and delta
    M_mean = torch.mean(M)
    # Calculate the dropout probability and cap it between 0 and 1
    dropout_prob = np.clip(10 / (nc * nt + eps), 0.0, 1.0)
    # Apply dropout with the capped probability
    M_drop = torch.nn.functional.dropout(M, p=dropout_prob)

    delta = M.max().detach()
    eff_lam = (lam / M_mean).detach()

    # Compute new distance matrix
    Mt = M
    row = delta * torch.ones((1, M.size(1))).to(X.device)
    col = torch.cat([delta * torch.ones((M.size(0), 1)).to(X.device), torch.zeros((1, 1)).to(X.device)], dim=0)
    Mt = torch.cat([M, row], dim=0)
    Mt = torch.cat([Mt, col], dim=1)

    # Compute marginal vectors
    a = torch.cat(
        [p * torch.ones((Xt.size(0), 1)).to(X.device) / (nt + 0.01), (1 - p) * torch.ones((1, 1)).to(X.device)],
        dim=0)
    b = torch.cat([(1 - p) * torch.ones((Xc.size(0), 1)).to(X.device) / nc, p * torch.ones((1, 1)).to(X.device)],
                  dim=0)

    # Compute kernel matrix
    Mlam = eff_lam * Mt
    K = torch.exp(-Mlam) + eps
    U = K * Mt
    ainvK = K / a

    u = a
    for i in range(its):
        u = 1.0 / (ainvK @ (b / (u.t() @ K).t()))

    v = b / (u.t() @ K).t()
    T = u * (v.t() * K)

    if not backpropT:
        T = T.detach()

    E = T * Mt
    D = 2 * torch.sum(E)

    return D, Mlam


def CFR_loss(t, e, y, y_, h_rep_norm,
             alpha: float,
             wass_iterations: int = 10,
             wass_lambda: float = 10.0,
             eps: float = 1e-6,
             wass_bpt: bool = True,
             return_items: bool = True):
    """
        Compute the Counterfactual Regression (CFR) loss.

        Parameters:
        ----------
        t : torch.Tensor
            Treatment indicators, where 1 indicates treatment and 0 indicates control.
        e : torch.Tensor
            Propensity scores, i.e., the probability of receiving treatment.
        y : torch.Tensor
            Observed outcomes.
        y_ : torch.Tensor
            Predicted outcomes.
        h_rep_norm : torch.Tensor
            Normalized representation of the input features (which is normalized e).
        wass_iterations : int
            Number of iterations for the Wasserstein distance calculation.
        wass_lambda : float
            Regularization parameter for the Wasserstein distance.
        wass_bpt : bool
            Flag indicating whether to backpropagate through the transport matrix in the Wasserstein distance calculation.
        eps: float
            Small value to ensure numerical stability.
        alpha : float
            Weighting factor for the imbalance error term.

        Returns:
        -------
        tot_error : torch.Tensor
            Total loss combining the factual loss and the imbalance error.

        Description:
        ------------
        This function computes the CFR loss, which is a combination of the factual loss (risk) and the imbalance error
        (imb_error). The factual loss measures the discrepancy between observed outcomes and predicted outcomes,
        weighted by sample weights. The imbalance error measures the distributional distance between treated and
        control groups in the representation space, using the Wasserstein distance.

        The function performs the following steps:
        1. Computes sample reweighting based on treatment indicators and propensity scores.
        2. Constructs the factual loss function using the reweighted squared errors between observed and predicted outcomes.
        3. Computes the imbalance error using the Wasserstein distance between treated and control groups in the
           representation space.
        4. Combines the factual loss and the imbalance error to obtain the total error.

        Example:
        --------
        tot_error = CFR_loss(t, e, y, y_, h_rep_norm, wass_iterations=50, wass_lambda=1, wass_bpt=False, alpha=0.1)
        """
    ''' Compute sample reweighting '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = t.to(device)
    e = e.to(device)
    y = y.to(device)
    y_ = y_.to(device)
    w_t = t * (1 - e) / e
    w_c = (1 - t) * e / (1 - e)
    sample_weight = torch.clamp(w_t + w_c, min=0.0, max=100.0)
    ''' Construct factual loss function '''
    risk = torch.mean(sample_weight * torch.sqrt(torch.abs(y - y_)))

    p_ipm = 0.5
    p_ipm = torch.tensor(p_ipm).to(device)
    ''' Compute imbalance error '''
    imb_dist, imb_mat = wasserstein(h_rep_norm, t, p_ipm,
                                    its=wass_iterations, lam=wass_lambda, backpropT=wass_bpt, eps=eps)
    imb_error = alpha * imb_dist

    ''' Total error '''
    loss_value = risk + imb_error

    if return_items:
        batch_items = {
            'loss': loss_value.item()
        }
        return loss_value, batch_items
    else:
        return loss_value


