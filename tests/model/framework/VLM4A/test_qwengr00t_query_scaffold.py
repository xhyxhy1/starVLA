import unittest

import torch

from starVLA.model.framework.VLM4A.QwenGR00TQueryScaffold import Qwen_GR00T_QueryScaffold


class DummyTokenizer:
    unk_token_id = 0

    def __init__(self):
        self.ids = {
            "<|skill_query|>": 101,
            "<|object_query|>": 102,
            "<|progress_query|>": 103,
        }

    def convert_tokens_to_ids(self, token):
        return self.ids[token]


def make_scaffold():
    scaffold = Qwen_GR00T_QueryScaffold.__new__(Qwen_GR00T_QueryScaffold)
    torch.nn.Module.__init__(scaffold)
    scaffold.query_token_names = ["<|skill_query|>", "<|object_query|>", "<|progress_query|>"]
    scaffold.query_token_ids = None
    scaffold.use_query_in_action = True
    scaffold.query_gate = torch.nn.Parameter(torch.tensor(-5.0))
    return scaffold


class QwenGR00TQueryScaffoldTest(unittest.TestCase):
    def test_extract_query_hidden_states_returns_tokens_in_configured_order(self):
        scaffold = make_scaffold()
        tokenizer = DummyTokenizer()
        hidden = torch.arange(2 * 6 * 4, dtype=torch.float32).reshape(2, 6, 4)
        input_ids = torch.tensor(
            [
                [11, 101, 12, 102, 13, 103],
                [21, 103, 101, 22, 102, 23],
            ]
        )

        query_hidden = scaffold._extract_query_hidden_states(hidden, input_ids, tokenizer)

        self.assertEqual(query_hidden.shape, (2, 3, 4))
        torch.testing.assert_close(query_hidden[0, 0], hidden[0, 1])
        torch.testing.assert_close(query_hidden[0, 1], hidden[0, 3])
        torch.testing.assert_close(query_hidden[0, 2], hidden[0, 5])
        torch.testing.assert_close(query_hidden[1, 0], hidden[1, 2])
        torch.testing.assert_close(query_hidden[1, 1], hidden[1, 4])
        torch.testing.assert_close(query_hidden[1, 2], hidden[1, 1])

    def test_build_action_condition_appends_gated_query_hidden(self):
        scaffold = make_scaffold()
        last_hidden = torch.ones(2, 4, 3)
        query_hidden = torch.full((2, 3, 3), 2.0)

        condition = scaffold._build_action_condition(last_hidden, query_hidden)

        gate = torch.sigmoid(scaffold.query_gate.detach()).to(query_hidden.dtype)
        self.assertEqual(condition.shape, (2, 7, 3))
        torch.testing.assert_close(condition[:, :4], last_hidden)
        torch.testing.assert_close(condition[:, 4:], gate * query_hidden)

    def test_query_diagnostics_are_scalar_tensors(self):
        scaffold = make_scaffold()
        query_hidden = torch.randn(2, 3, 4)

        diagnostics = scaffold._query_diagnostics(query_hidden)

        self.assertEqual(set(diagnostics), {"query_gate", "query_hidden_norm", "query_pairwise_cosine"})
        self.assertEqual(diagnostics["query_gate"].shape, ())
        self.assertEqual(diagnostics["query_hidden_norm"].shape, ())
        self.assertEqual(diagnostics["query_pairwise_cosine"].shape, ())
