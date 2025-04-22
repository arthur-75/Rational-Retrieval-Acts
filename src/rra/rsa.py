import torch
from tqdm import tqdm


class RSA:
    def __init__(
        self, alpha=1.0, iterations=1, device=None, batch_row=1, classic=True, lambda_=1
    ):

        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else device
        )
        self.alpha = torch.tensor(alpha, device=self.device)
        self.iterations = iterations
        self.classic = classic
        self.batch_row = batch_row
        self.lambda_ = lambda_

    def soft_max(
        self,
        lexicon,
        dim=None,
        Vd=None,
        Ut=None,
        speaker_=False,
        cost=torch.tensor(0),
        classic=False,
    ):
        lexicon = lexicon.coalesce()
        # Get indices and values
        cost = cost.to(device=self.device)

        indices = lexicon.indices().to(device=self.device)
        values = lexicon.values().to(device=self.device)
        if speaker_:
            values = (torch.log(values) * self.alpha) - cost

        if classic:
            exp_values = values + self.lambda_  # .to( dtype=torch.int32)
        else:
            exp_values = torch.exp(values).to(device=self.device)

        size = lexicon.size()
        if Vd is None:
            Vd = torch.ones(size[1], device=self.device) * self.lambda_
        size_dim_sum = dim
        dim = abs(dim - 1)
        sum_exp = torch.zeros(size[dim], device=self.device)
        sum_exp.scatter_add_(0, indices[dim], exp_values)

        #########################################

        if speaker_:

            if self.batch_row is None:
                mask = torch.ones(size[0], size[1], dtype=torch.bool)

                # Set existing pairs to False in the mask
                mask[indices[0], indices[1]] = False

                # Get indices where mask is True (zero positions)
                zero_i, zero_j = torch.where(mask)

                # Vectorized computation
                # Expand dimensions for broadcasting
                Vd_expanded = Vd[zero_j]  # Shape: [num_zeros]
                Ut_expanded = Ut[zero_i]  # Shape: [num_zeros]

                # Compute all exp values at once
                exp_values_zero = torch.exp(
                    self.alpha * torch.log(Vd_expanded * Ut_expanded)
                )
                # Use scatter_add to sum values for each j
                sum_exp.scatter_add_(0, zero_j, exp_values_zero)
            else:
                mask = torch.ones(
                    size[0], size[1], dtype=torch.bool, device=self.device
                )
                mask[indices[0], indices[1]] = False

                log_Vd = torch.log(Vd)
                log_Ut = torch.log(Ut)
                print("Pragmatic Speaker...")
                # Process in batches
                for start_batch in tqdm(range(0, size[0], self.batch_row)):
                    # Define batch boundaries
                    end_batch = min(start_batch + self.batch_row, size[0])

                    # Get current batch of mask
                    batch_mask = mask[start_batch:end_batch]

                    # Find non-zero positions in current batch
                    batch_zero_i, batch_zero_j = torch.where(batch_mask)

                    if len(batch_zero_i) == 0:
                        continue

                    # Adjust indices to account for batching
                    batch_zero_i = batch_zero_i + start_batch

                    # Compute log values for current batch
                    batch_log_values = log_Vd[batch_zero_j] + log_Ut[batch_zero_i]

                    # Compute exp values with scaling factor alpha
                    exp_values_batch = torch.exp(self.alpha * batch_log_values)

                    # Update sum_exp using scatter_add_
                    sum_exp.scatter_add_(0, batch_zero_j, exp_values_batch)

                # Final computations

            Vd = torch.exp(self.alpha * log_Vd) / sum_exp
            Ut = torch.exp(self.alpha * log_Ut - cost)

            # Count non-zero elements per row/column
        else:
            nonzero_count_per_row = torch.bincount(indices[dim], minlength=size[dim])
            # Compute the number of zeros in each row
            zero_count = size[size_dim_sum] - nonzero_count_per_row
            sum_exp += zero_count * self.lambda_
            Ut = self.lambda_ / sum_exp

        normalized_values = exp_values / sum_exp[indices[dim]].to(device=self.device)

        # Create the normalized sparse tensor
        normalized_lexicon = torch.sparse_coo_tensor(
            indices, normalized_values, size, device=self.device
        )

        return normalized_lexicon, Vd, Ut

    def literal_listener(self, lexicon):
        print("Literal Pistener...")
        return self.soft_max(lexicon, dim=1, classic=self.classic)

    def speaker(self, listener, Vd, Ut, cost=0):
        """Speaker reasoning based on the listener: P(m | w), with batching."""
        # Softmax to get speaker probabilities
        speaker_distribution_batch, Vd, Ut = self.soft_max(
            listener, 0, Vd, Ut, speaker_=True, classic=False
        )

        return speaker_distribution_batch, Vd, Ut

    def pragmatic_listener(self, speaker, Vd, Ut):
        """Pragmatic listener updated from the speaker: P(w | m), with batching."""
        speaker = speaker.coalesce()
        # listener_distribution_batch = F.normalize(speaker_distribution, p=self.norm, dim=1)
        dim = 1
        indices = speaker.indices()
        values = speaker.values()
        size_dim_sum = dim
        dim = abs(dim - 1)
        size = speaker.size()
        # Compute the norm for each row
        row_sums = torch.zeros(size[dim], device=self.device)
        row_sums.scatter_add_(0, indices[dim], values.abs())
        if self.batch_row is None:
            mask = torch.ones(size[0], size[1], dtype=torch.bool, device=self.device)

            # Set existing pairs to False in the mask
            mask[indices[0], indices[1]] = False

            # Get indices where mask is True (zero positions)
            zero_i, zero_j = torch.where(mask)

            # Vectorized computation
            # Expand dimensions for broadcasting
            Vd_expanded = Vd[zero_j]  # Shape: [num_zeros]
            Ut_expanded = Ut[zero_i]  # Shape: [num_zeros]

            # Compute all exp values at once
            exp_values_zero = Vd_expanded * Ut_expanded
            # Use scatter_add to sum values for each j
            row_sums.scatter_add_(0, zero_i, exp_values_zero)

        else:
            mask = torch.ones(size[0], size[1], dtype=torch.bool, device=self.device)
            mask[indices[0], indices[1]] = False

            # Process in batches
            print("Pragmatic Listener...")
            for start_batch in tqdm(range(0, size[0], self.batch_row)):
                # Define batch boundaries
                end_batch = min(start_batch + self.batch_row, size[0])

                # Get current batch of mask
                batch_mask = mask[start_batch:end_batch]

                # Find non-zero positions in current batch
                batch_zero_i, batch_zero_j = torch.where(batch_mask)

                if len(batch_zero_i) == 0:
                    continue

                # Adjust indices to account for batching
                batch_zero_i = batch_zero_i + start_batch

                # Compute values for current batch
                batch_values = Vd[batch_zero_j] * Ut[batch_zero_i]

                # Update row sums
                row_sums.scatter_add_(0, batch_zero_i, batch_values)

        Ut = Ut / row_sums
        normalized_values = values / row_sums.to(device=self.device)[indices[dim]]
        normalized_values = torch.sparse_coo_tensor(
            indices, normalized_values, speaker.shape, device=self.device
        )
        return normalized_values, Vd, Ut

    # Create a new sparse tensor with normalized values

    def run(self, lexion_):
        """Run the RSA reasoning iteratively."""
        listener_distribution, Vd, Ut = self.literal_listener(lexion_)
        del lexion_
        cost = None
        for i in range(self.iterations):
            speaker_distribution, Vd, Ut = self.speaker(listener_distribution, Vd, Ut)
            del listener_distribution
            listener_distribution, Vd, Ut = self.pragmatic_listener(
                speaker_distribution, Vd, Ut
            )
            del speaker_distribution

        return listener_distribution, Vd, Ut
