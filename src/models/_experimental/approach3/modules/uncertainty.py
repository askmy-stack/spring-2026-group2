"""
Uncertainty Quantification Module
---------------------------------
Methods for estimating prediction uncertainty in seizure detection.

Critical for clinical deployment:
- Flag low-confidence predictions for human review
- Distinguish between data and model uncertainty
- Improve trust in AI-assisted diagnosis

References:
- Dropout as Bayesian Approximation (Gal & Ghahramani, 2016)
- Evidential Deep Learning (Sensoy et al., 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict


class MCDropout(nn.Module):
    """
    Monte Carlo Dropout wrapper.
    Keeps dropout active during inference for uncertainty estimation.
    """

    def __init__(self, p: float = 0.2):
        super().__init__()
        self.p = p
        self.dropout = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Always apply dropout (even in eval mode)
        return F.dropout(x, p=self.p, training=True)


def mc_dropout_inference(
    model: nn.Module,
    x: torch.Tensor,
    n_samples: int = 30,
    dropout_layers: Optional[list] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform Monte Carlo Dropout inference.

    Args:
        model: Neural network with dropout layers
        x: Input tensor
        n_samples: Number of forward passes
        dropout_layers: Optional list of dropout layer names to keep active

    Returns:
        mean_pred: Mean prediction across samples
        std_pred: Standard deviation (uncertainty)
        all_preds: All predictions (n_samples, batch, ...)
    """
    model.train()  # Enable dropout

    predictions = []
    with torch.no_grad():
        for _ in range(n_samples):
            output = model(x)
            if isinstance(output, tuple):
                output = output[0]
            pred = torch.sigmoid(output)
            predictions.append(pred)

    predictions = torch.stack(predictions, dim=0)  # (n_samples, batch, 1)

    mean_pred = predictions.mean(dim=0)
    std_pred = predictions.std(dim=0)

    return mean_pred, std_pred, predictions


def compute_uncertainty(
    predictions: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Compute various uncertainty metrics from MC samples.

    Args:
        predictions: MC samples (n_samples, batch, 1)

    Returns:
        Dictionary with uncertainty metrics
    """
    n_samples = predictions.shape[0]

    # Mean prediction
    mean = predictions.mean(dim=0)

    # Predictive uncertainty (total)
    predictive_std = predictions.std(dim=0)

    # Aleatoric uncertainty (data uncertainty)
    # Estimated as mean of per-sample variances
    aleatoric = (predictions * (1 - predictions)).mean(dim=0)

    # Epistemic uncertainty (model uncertainty)
    # Difference between total and aleatoric
    epistemic = predictive_std ** 2 - aleatoric
    epistemic = torch.clamp(epistemic, min=0)  # Ensure non-negative

    # Entropy
    mean_clipped = torch.clamp(mean, 1e-7, 1 - 1e-7)
    entropy = -(mean_clipped * torch.log(mean_clipped) +
                (1 - mean_clipped) * torch.log(1 - mean_clipped))

    # Mutual information (approximation)
    sample_entropies = -(predictions * torch.log(predictions + 1e-7) +
                         (1 - predictions) * torch.log(1 - predictions + 1e-7))
    mutual_info = entropy - sample_entropies.mean(dim=0)

    return {
        "mean": mean,
        "predictive_std": predictive_std,
        "aleatoric": aleatoric,
        "epistemic": torch.sqrt(epistemic),
        "entropy": entropy,
        "mutual_info": mutual_info,
    }


class EvidentialClassifier(nn.Module):
    """
    Evidential Deep Learning classifier.
    Outputs Dirichlet distribution parameters for principled uncertainty.

    Instead of point predictions, outputs evidence for each class.
    Uncertainty is quantified as inverse of total evidence.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input features (batch, input_dim)

        Returns:
            prob: Predicted probabilities (batch, num_classes)
            uncertainty: Uncertainty estimate (batch, 1)
            evidence: Evidence for each class (batch, num_classes)
        """
        # Get evidence (non-negative)
        evidence = F.softplus(self.net(x))

        # Dirichlet parameters
        alpha = evidence + 1

        # Total evidence (Dirichlet strength)
        S = alpha.sum(dim=-1, keepdim=True)

        # Predicted probability
        prob = alpha / S

        # Uncertainty = inverse of evidence
        uncertainty = self.num_classes / S

        return prob, uncertainty, evidence

    def get_prediction(self, x: torch.Tensor) -> torch.Tensor:
        """Get class prediction."""
        prob, _, _ = self.forward(x)
        return prob[:, 1:2]  # Return seizure probability


class EvidentialLoss(nn.Module):
    """
    Loss function for evidential deep learning.
    Combines cross-entropy with KL divergence regularization.
    """

    def __init__(self, num_classes: int = 2, annealing_step: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.annealing_step = annealing_step

    def kl_divergence(self, alpha: torch.Tensor) -> torch.Tensor:
        """KL divergence from Dirichlet to uniform."""
        ones = torch.ones_like(alpha)
        sum_alpha = alpha.sum(dim=-1, keepdim=True)
        sum_ones = ones.sum(dim=-1, keepdim=True)

        kl = (
            torch.lgamma(sum_alpha) - torch.lgamma(sum_ones)
            - (torch.lgamma(alpha) - torch.lgamma(ones)).sum(dim=-1, keepdim=True)
            + ((alpha - ones) * (torch.digamma(alpha) - torch.digamma(sum_alpha))).sum(
                dim=-1, keepdim=True
            )
        )
        return kl.mean()

    def forward(
        self,
        evidence: torch.Tensor,
        targets: torch.Tensor,
        epoch: int = 0,
    ) -> torch.Tensor:
        """
        Compute evidential loss.

        Args:
            evidence: Evidence (batch, num_classes)
            targets: Class labels (batch,)
            epoch: Current epoch for annealing

        Returns:
            Total loss
        """
        alpha = evidence + 1
        S = alpha.sum(dim=-1, keepdim=True)

        # One-hot targets
        if targets.dim() == 1:
            y_onehot = F.one_hot(targets.long(), self.num_classes).float()
        else:
            y_onehot = targets

        # Cross-entropy loss
        ce_loss = (y_onehot * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=-1).mean()

        # KL regularization (annealed)
        annealing_coef = min(1.0, epoch / self.annealing_step)

        # Remove evidence for correct class before KL
        alpha_tilde = y_onehot + (1 - y_onehot) * alpha
        kl_loss = annealing_coef * self.kl_divergence(alpha_tilde)

        return ce_loss + kl_loss


class UncertaintyWrapper(nn.Module):
    """
    Wrapper that adds uncertainty estimation to any model.
    Combines MC Dropout with optional evidential output.
    """

    def __init__(
        self,
        base_model: nn.Module,
        use_evidential: bool = False,
        mc_dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.base_model = base_model
        self.use_evidential = use_evidential
        self.mc_dropout = MCDropout(mc_dropout_rate)

        if use_evidential:
            # Add evidential head
            # Assumes base model outputs features
            self.evidential_head = EvidentialClassifier(
                input_dim=128,  # Adjust based on base model
                num_classes=2,
            )

    def forward(
        self, x: torch.Tensor, return_uncertainty: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass with optional uncertainty.

        Args:
            x: Input tensor
            return_uncertainty: Whether to compute uncertainty

        Returns:
            logits: Model predictions
            uncertainty: (optional) Uncertainty estimate
        """
        output = self.base_model(x)

        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        if not return_uncertainty:
            return logits

        if self.use_evidential:
            # Use evidential uncertainty
            prob, uncertainty, _ = self.evidential_head(logits)
            return prob[:, 1:2], uncertainty
        else:
            # Use prediction confidence as simple uncertainty
            prob = torch.sigmoid(logits)
            uncertainty = 1 - torch.abs(2 * prob - 1)  # Max at 0.5
            return logits, uncertainty


class ConfidenceCalibration(nn.Module):
    """
    Temperature scaling for confidence calibration.
    Ensures predicted probabilities match empirical accuracy.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> None:
        """Optimize temperature on validation set."""
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled = self.forward(logits)
            loss = F.binary_cross_entropy_with_logits(scaled, labels.float())
            loss.backward()
            return loss

        optimizer.step(closure)


def flag_uncertain_predictions(
    predictions: torch.Tensor,
    uncertainties: torch.Tensor,
    threshold: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flag predictions with high uncertainty for human review.

    Args:
        predictions: Model predictions
        uncertainties: Uncertainty estimates
        threshold: Uncertainty threshold

    Returns:
        flagged: Boolean mask of flagged predictions
        confidence_scores: 1 - uncertainty
    """
    flagged = uncertainties > threshold
    confidence_scores = 1 - uncertainties

    return flagged, confidence_scores


if __name__ == "__main__":
    # Test MC Dropout
    print("Testing MC Dropout...")

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 1),
            )

        def forward(self, x):
            return self.fc(x)

    model = SimpleModel()
    x = torch.randn(8, 256)

    mean, std, preds = mc_dropout_inference(model, x, n_samples=30)
    print(f"Mean shape: {mean.shape}, Std shape: {std.shape}")
    print(f"Mean uncertainty: {std.mean().item():.4f}")

    # Test Evidential
    print("\nTesting Evidential Classifier...")
    evidential = EvidentialClassifier(input_dim=256)
    prob, unc, evidence = evidential(x)
    print(f"Prob shape: {prob.shape}, Uncertainty shape: {unc.shape}")
    print(f"Mean uncertainty: {unc.mean().item():.4f}")
