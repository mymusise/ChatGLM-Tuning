import torch

from trl.trainer.ppo_trainer import (
    logprobs_from_logits,
    PreTrainedModelWrapper,
    PPODecorators,
    PPOTrainer,
)


class ChatGLMPPOTrainer(PPOTrainer):
    def pad_input_ids(self, ids, max_length):
        # _ids = torch.ones(max_length, dtype=torch.int8) * self.model.config.pad_token_id
        _ids = torch.ones(max_length, dtype=torch.int64) * 3
        _ids[: len(ids)] = ids
        return _ids

    def prepare_model_inputs(self, queries: torch.Tensor, responses: torch.Tensor):
        input_ids = [torch.cat([q, r]) for q, r in zip(queries, responses)]
        max_length = max([len(ids) for ids in input_ids])
        input_data = self.data_collator(
            [{"input_ids": self.pad_input_ids(ids, max_length)} for ids in input_ids]
        ).to(self.current_device)

        input_data.pop("labels", None)  # we don't want to compute LM losses

        return input_data

    @PPODecorators.empty_cuda_cache()
    def batched_forward_pass(
        self,
        model: PreTrainedModelWrapper,
        queries: torch.Tensor,
        responses: torch.Tensor,
        model_inputs: dict,
    ):
        """
        Calculate model outputs in multiple batches.

        Args:
            queries (`torch.LongTensor`):
                List of tensors containing the encoded queries, shape (`batch_size`, `query_length`)
            responses (`torch.LongTensor`):
                List of tensors containing the encoded responses, shape (`batch_size`, `response_length`)
        Returns:
            (tuple):
                - all_logprobs (`torch.FloatTensor`): Log probabilities of the responses,
                    shape (`batch_size`, `response_length`)
                - all_ref_logprobs (`torch.FloatTensor`): Log probabilities of the responses,
                    shape (`batch_size`, `response_length`)
                - all_values (`torch.FloatTensor`): Values of the responses, shape (`batch_size`, `response_length`)
        """
        bs = len(queries)
        fbs = self.config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        for i in range(int(bs / fbs)):
            input_kwargs = {
                key: value[i * fbs : (i + 1) * fbs]
                for key, value in model_inputs.items()
            }
            logits, _, values = model(**input_kwargs)
            input_ids = input_kwargs["input_ids"]

            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            masks = torch.zeros_like(input_ids)
            for i, ids in enumerate(input_ids):
                start_i = (ids == self.model.config.bos_token_id).nonzero()
                end_i = (ids == self.model.config.eos_token_id).nonzero()
                if len(end_i):
                    end_i = end_i[0][0] + 1
                else:
                    end_i = None
                masks[i][start_i:end_i] = 1

            all_logits.append(logits)
            all_values.append(values.rot90())
            all_logprobs.append(logprobs)
            all_masks.append(masks)

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1],
            torch.cat(all_values)[:, :-1],
            torch.cat(all_masks)[:, :-1],
        )

