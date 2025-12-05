from torch.utils.data import WeightedRandomSampler

def make_weighted_sampler(df_split):
    class_counts = df_split["label_idx"].value_counts().sort_index().values.astype(float)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[df_split["label_idx"].values]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
