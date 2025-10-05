import numpy as np
import torch
from scipy import stats
from typing import Dict, Union


def compute_entropy_statistics(entropy_tensor: Union[torch.Tensor, np.ndarray], 
                              response_mask: Union[torch.Tensor, np.ndarray] = None,
                              chunk_size: int = 1024) -> Dict[str, float]:
    """
    Compute comprehensive statistics for entropy sequences using vectorized operations with chunked processing.
    
    Args:
        entropy_tensor: Shape (B, N) where B=batch_size, N=sequence_length
                       Values typically in range [0, 3] with most near 0
        response_mask: Shape (B, N) where 1=valid token, 0=padding/invalid
                      If None, all tokens are considered valid
        chunk_size: Batch size for each chunk to process (default: 1024)
    
    Returns:
        Dictionary with mean statistics across the batch
    """
    # Convert to torch tensor for vectorized operations
    if isinstance(entropy_tensor, np.ndarray):
        entropy_data = torch.from_numpy(entropy_tensor).float()
    else:
        entropy_data = entropy_tensor.float()
    
    if response_mask is not None:
        if isinstance(response_mask, np.ndarray):
            mask_data = torch.from_numpy(response_mask).float()
        else:
            mask_data = response_mask.float()
    else:
        # If no mask provided, all tokens are valid
        mask_data = torch.ones_like(entropy_data)
    
    B, N = entropy_data.shape
    assert mask_data.shape == entropy_data.shape, "Response mask must have same shape as entropy tensor"
    
    # Process data in chunks to reduce memory usage
    all_means = []
    all_variances = []
    all_skewness = []
    all_kurtosis = []
    
    for start_idx in range(0, B, chunk_size):
        end_idx = min(start_idx + chunk_size, B)
        
        # Extract chunk
        chunk_entropy = entropy_data[start_idx:end_idx]  # Shape: (chunk_B, N)
        chunk_mask = mask_data[start_idx:end_idx]        # Shape: (chunk_B, N)
        chunk_B = chunk_entropy.shape[0]
        
        # Count valid tokens per sequence in chunk
        valid_counts = chunk_mask.sum(dim=1)  # Shape: (chunk_B,)
        
        # Filter out sequences with insufficient valid tokens
        valid_seqs = valid_counts >= 5  # Boolean mask for sequences with enough data
        
        if not valid_seqs.any():
            # Skip this chunk if no valid sequences
            continue
        
        # Apply mask to entropy data
        masked_entropy = chunk_entropy * chunk_mask  # Zero out invalid tokens
        
        # Compute means for each sequence
        means = masked_entropy.sum(dim=1) / valid_counts.clamp(min=1)  # Shape: (chunk_B,)
        
        # Center the data for variance/skewness/kurtosis computation
        centered = (chunk_entropy - means.unsqueeze(1)) * chunk_mask  # Shape: (chunk_B, N)
        
        # Compute moments
        valid_counts_clamped = valid_counts.clamp(min=1)
        
        # Variance (using ddof=1 equivalent: divide by n-1)
        second_moment = (centered ** 2).sum(dim=1) / (valid_counts_clamped - 1).clamp(min=1)
        variance = second_moment
        
        # Standard deviation
        std = variance.sqrt()
        
        # Third and fourth moments for skewness and kurtosis
        third_moment = (centered ** 3).sum(dim=1) / valid_counts_clamped
        fourth_moment = (centered ** 4).sum(dim=1) / valid_counts_clamped
        
        # Skewness: E[(X-μ)³] / σ³
        skewness = torch.full((chunk_B,), float('nan'), device=chunk_entropy.device)
        valid_std_mask = (std > 1e-8) & valid_seqs
        skewness[valid_std_mask] = third_moment[valid_std_mask] / (std[valid_std_mask] ** 3)
        skewness[valid_seqs & (std <= 1e-8)] = 0.0  # Zero skewness for constant sequences
        
        # Kurtosis: E[(X-μ)⁴] / σ⁴ - 3 (excess kurtosis)
        kurtosis = torch.full((chunk_B,), float('nan'), device=chunk_entropy.device)
        kurtosis[valid_std_mask] = fourth_moment[valid_std_mask] / (std[valid_std_mask] ** 4) - 3.0
        kurtosis[valid_seqs & (std <= 1e-8)] = 0.0  # Zero excess kurtosis for constant sequences
        
        # Collect valid results from this chunk
        valid_means_chunk = means[valid_seqs]
        valid_variance_chunk = variance[valid_seqs]
        valid_skewness_chunk = skewness[valid_seqs]
        valid_kurtosis_chunk = kurtosis[valid_seqs]
        
        # Append to global lists
        if len(valid_means_chunk) > 0:
            all_means.append(valid_means_chunk)
            all_variances.append(valid_variance_chunk)
            all_skewness.append(valid_skewness_chunk)
            all_kurtosis.append(valid_kurtosis_chunk)
    
    # Check if we have any valid results
    if len(all_means) == 0:
        return {'mean': float('nan'), 'variance': float('nan'), 
                'skewness': float('nan'), 'kurtosis': float('nan')}
    
    # Concatenate all chunks
    all_means = torch.cat(all_means, dim=0)
    all_variances = torch.cat(all_variances, dim=0)
    all_skewness = torch.cat(all_skewness, dim=0)
    all_kurtosis = torch.cat(all_kurtosis, dim=0)
    
    # Filter out any remaining NaN values and compute batch statistics
    def safe_mean(tensor):
        finite_mask = torch.isfinite(tensor)
        if finite_mask.any():
            return tensor[finite_mask].mean().item()
        else:
            return float('nan')
    
    batch_stats = {
        'mean': safe_mean(all_means),
        'variance': safe_mean(all_variances),
        'skewness': safe_mean(all_skewness),
        'kurtosis': safe_mean(all_kurtosis)
    }
    
    return batch_stats

def compute_entropy_statistics_old(entropy_tensor: Union[torch.Tensor, np.ndarray], 
                              response_mask: Union[torch.Tensor, np.ndarray] = None) -> Dict[str, float]:
    """
    Compute comprehensive statistics for entropy sequences.
    
    Args:
        entropy_tensor: Shape (B, N) where B=batch_size, N=sequence_length
                       Values typically in range [0, 3] with most near 0
        response_mask: Shape (B, N) where 1=valid token, 0=padding/invalid
                      If None, all tokens are considered valid
    
    Returns:
        Dictionary with mean statistics across the batch
    """
    # Convert to numpy if torch tensor
    if isinstance(entropy_tensor, torch.Tensor):
        entropy_data = entropy_tensor.detach().cpu().numpy()
    else:
        entropy_data = entropy_tensor
    
    if response_mask is not None:
        if isinstance(response_mask, torch.Tensor):
            mask_data = response_mask.detach().cpu().numpy()
        else:
            mask_data = response_mask
    else:
        # If no mask provided, all tokens are valid
        mask_data = np.ones_like(entropy_data)
    
    B, N = entropy_data.shape
    assert mask_data.shape == entropy_data.shape, "Response mask must have same shape as entropy tensor"
    
    # Initialize results storage
    results = {
        'mean': [], 'variance': [], 'skewness': [], 'kurtosis': [],
        'change_point_freq': [],
        'ac_lag1': [], 'ac_lag5': [], 'ac_lag10': [],
        'decay_rate': [], 'ac_sum': [], 'memory_length': []
    }
    
    for i in range(B):
        # Extract valid tokens only (where mask == 1)
        valid_mask = mask_data[i] == 1
        sequence = entropy_data[i][valid_mask]
        
        # Skip sequences that are too short after masking
        if len(sequence) < 5:  # Need minimum length for meaningful statistics
            for key in results.keys():
                results[key].append(None)
            continue
        
        # Basic statistics
        results['mean'].append(np.mean(sequence))
        results['variance'].append(np.var(sequence, ddof=1))
        results['skewness'].append(stats.skew(sequence))
        results['kurtosis'].append(stats.kurtosis(sequence, fisher=True))  # Excess kurtosis (kurt-3)
        
        # Change point frequency
        results['change_point_freq'].append(_compute_change_point_frequency(sequence))
        
        # Autocorrelation features
        ac_features = _compute_autocorr_features(sequence)
        for key in ['ac_lag1', 'ac_lag5', 'ac_lag10', 'decay_rate', 'ac_sum', 'memory_length']:
            results[key].append(ac_features[key])
    
    # Compute mean across batch
    batch_stats = {}
    for key, values in results.items():
        # Filter out None values for robust averaging
        valid_values = [v for v in values if v is not None and not np.isnan(v)]
        if valid_values:
            batch_stats[key] = np.mean(valid_values)
        else:
            batch_stats[key] = np.nan
    
    # # Compute higher-order moments for skewness and kurtosis sequences
    # # Use existing results directly
    # valid_skewness = [v for v in results['skewness'] if v is not None and not np.isnan(v)]
    # if len(valid_skewness) > 0:
    #     skewness_array = np.array(valid_skewness)
    #     batch_stats['skewness_mean'] = np.mean(skewness_array)
    #     batch_stats['skewness_var'] = np.var(skewness_array, ddof=1)
    #     batch_stats['skewness_skewness'] = stats.skew(skewness_array)
    #     batch_stats['skewness_kurtosis'] = stats.kurtosis(skewness_array, fisher=True)
    # else:
    #     batch_stats['skewness_mean'] = np.nan
    #     batch_stats['skewness_var'] = np.nan
    #     batch_stats['skewness_skewness'] = np.nan
    #     batch_stats['skewness_kurtosis'] = np.nan
    
    # valid_kurtosis = [v for v in results['kurtosis'] if v is not None and not np.isnan(v)]
    # if len(valid_kurtosis) > 0:
    #     kurtosis_array = np.array(valid_kurtosis)
    #     batch_stats['kurtosis_mean'] = np.mean(kurtosis_array)
    #     batch_stats['kurtosis_var'] = np.var(kurtosis_array, ddof=1)
    #     batch_stats['kurtosis_skewness'] = stats.skew(kurtosis_array)
    #     batch_stats['kurtosis_kurtosis'] = stats.kurtosis(kurtosis_array, fisher=True)
    # else:
    #     batch_stats['kurtosis_mean'] = np.nan
    #     batch_stats['kurtosis_var'] = np.nan
    #     batch_stats['kurtosis_skewness'] = np.nan
    #     batch_stats['kurtosis_kurtosis'] = np.nan
    
    return batch_stats


def _compute_change_point_frequency(sequence: np.ndarray, 
                                  window_size: int = 50, 
                                  threshold_factor: float = 1.5) -> float:
    """
    Compute change point frequency using sliding window variance method.
    
    Args:
        sequence: 1D entropy sequence
        window_size: Size of sliding window for variance computation
        threshold_factor: Multiplier for variance change threshold
    """
    if len(sequence) < 2 * window_size:
        return 0.0
    
    # Compute variance in sliding windows
    window_vars = []
    for i in range(len(sequence) - window_size + 1):
        window = sequence[i:i + window_size]
        window_vars.append(np.var(window, ddof=1))
    
    window_vars = np.array(window_vars)
    
    # Detect change points based on variance changes
    if len(window_vars) < 2:
        return 0.0
    
    # Threshold based on overall variance change distribution
    var_changes = np.abs(np.diff(window_vars))
    if len(var_changes) == 0:
        return 0.0
    
    threshold = threshold_factor * np.std(var_changes)
    change_points = np.sum(var_changes > threshold)
    
    # Normalize by number of valid windows (not sequence length)
    return change_points / len(window_vars)


def _compute_autocorr_features(sequence: np.ndarray, max_lag: int = 50) -> Dict[str, Union[float, None]]:
    """
    Compute various autocorrelation-based features.
    """
    if len(sequence) < 11:  # Need at least 11 points for lag-10
        return {
            'ac_lag1': None, 'ac_lag5': None, 'ac_lag10': None,
            'decay_rate': None, 'ac_sum': None, 'memory_length': None
        }
    
    # Remove mean for autocorrelation computation
    sequence_centered = sequence - np.mean(sequence)
    
    def autocorr_at_lag(data, lag):
        """Compute autocorrelation at specific lag with better error handling"""
        if len(data) <= lag:
            return 0.0
        try:
            corr = np.corrcoef(data[:-lag], data[lag:])[0, 1]
            return corr if not np.isnan(corr) else 0.0
        except:
            return 0.0
    
    # Specific lag autocorrelations
    ac_lag1 = autocorr_at_lag(sequence_centered, 1)
    ac_lag5 = autocorr_at_lag(sequence_centered, 5) if len(sequence) > 5 else 0.0
    ac_lag10 = autocorr_at_lag(sequence_centered, 10) if len(sequence) > 10 else 0.0
    
    # Compute autocorrelations up to max_lag
    max_computable_lag = min(max_lag, len(sequence) // 2)
    autocorrs = []
    
    for lag in range(1, max_computable_lag + 1):
        ac = autocorr_at_lag(sequence_centered, lag)
        autocorrs.append(ac)
    
    # Decay rate (exponential decay fitting)
    decay_rate = 0.0
    if len(autocorrs) > 0 and abs(ac_lag1) > 0.01:  # Only if there's meaningful correlation
        # Find where |autocorr| drops below 0.1
        decay_lag = None
        for i, ac in enumerate(autocorrs):
            if abs(ac) < 0.1:
                decay_lag = i + 1
                break
        
        if decay_lag is not None and decay_lag > 1:
            decay_rate = -np.log(max(abs(autocorrs[decay_lag-1]), 1e-10)) / decay_lag
    
    # AC sum (sum of positive autocorrelations)
    ac_sum = sum(max(0, ac) for ac in autocorrs)
    
    # Memory length (first lag where |ρ(k)| < 1/e ≈ 0.37)
    memory_threshold = 1.0 / np.e  # ≈ 0.37
    memory_length = max_computable_lag  # Default to max lag
    for i, ac in enumerate(autocorrs):
        if abs(ac) < memory_threshold:
            memory_length = i + 1
            break
    
    return {
        'ac_lag1': ac_lag1,
        'ac_lag5': ac_lag5,
        'ac_lag10': ac_lag10,
        'decay_rate': decay_rate,
        'ac_sum': ac_sum,
        'memory_length': memory_length
    }


def compute_kurtosis_vectorized(entropy_tensor: torch.Tensor, 
                               response_mask: torch.Tensor = None) -> torch.Tensor:
    """Vectorized kurtosis computation using PyTorch operations."""
    B = entropy_tensor.shape[0]
    mask = response_mask.float() if response_mask is not None else torch.ones_like(entropy_tensor)
    valid_counts = mask.sum(dim=1)
    
    kurtosis_values = torch.full((B,), float('nan'), device=entropy_tensor.device)
    valid_seqs = valid_counts >= 5
    
    if not valid_seqs.any():
        return kurtosis_values
    
    # Compute masked statistics
    masked_entropy = entropy_tensor * mask
    means = (masked_entropy.sum(dim=1) / valid_counts.clamp(min=1)).unsqueeze(1)
    centered = (entropy_tensor - means) * mask
    
    # Compute moments and kurtosis
    counts_clamped = valid_counts.clamp(min=1)
    second_moment = (centered ** 2).sum(dim=1) / counts_clamped
    fourth_moment = (centered ** 4).sum(dim=1) / counts_clamped
    std = second_moment.sqrt()
    
    # Assign kurtosis values
    valid_with_std = valid_seqs & (std > 1e-8)
    kurtosis_values[valid_with_std] = fourth_moment[valid_with_std] / (std[valid_with_std] ** 4) - 3.0
    kurtosis_values[valid_seqs & (std <= 1e-8)] = 0.0
    
    return kurtosis_values


# Example usage and testing
if __name__ == "__main__":
    # Create example entropy data (most values near 0, some larger)
    B, N = 4, 1000
    
    # Simulate entropy data: mostly small values with occasional spikes
    np.random.seed(42)
    entropy_data = []
    response_masks = []
    
    for _ in range(B):
        # Base entropy (mostly near 0)
        base_entropy = np.random.exponential(0.1, N)
        
        # Add some larger spikes (simulate high-uncertainty tokens)
        spike_indices = np.random.choice(N, size=N//20, replace=False)
        base_entropy[spike_indices] += np.random.uniform(1.0, 3.0, len(spike_indices))
        
        # Create response mask (simulate some padding/invalid tokens)
        valid_length = np.random.randint(N//2, N)  # Random valid length between 50%-100%
        mask = np.zeros(N)
        mask[:valid_length] = 1
        
        entropy_data.append(base_entropy)
        response_masks.append(mask)
    
    entropy_tensor = np.array(entropy_data)
    response_mask = np.array(response_masks)
    
    # Compute statistics with mask
    stats_result = compute_entropy_statistics(entropy_tensor, response_mask)
    
    # Display results
    print("Entropy Statistics (averaged across batch, valid tokens only):")
    print("=" * 60)
    for key, value in stats_result.items():
        if not np.isnan(value):
            print(f"{key:18s}: {value:.6f}")
        else:
            print(f"{key:18s}: NaN (insufficient valid data)")
    
    # Example without mask (all tokens valid)
    print("\n" + "="*60)
    print("For comparison - without mask (all tokens):")
    stats_no_mask = compute_entropy_statistics(entropy_tensor)
    for key, value in stats_no_mask.items():
        print(f"{key:18s}: {value:.6f}")