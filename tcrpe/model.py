import unittest
import torch
from tape import TAPETokenizer
from tape.models.modeling_bert import ProteinBertAbstractModel, ProteinBertModel
from tape.models.modeling_utils import SimpleMLP

class BertEpitopeClassification(ProteinBertAbstractModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = ProteinBertModel(config)
        self.classifier = SimpleMLP(config.hidden_size, 512, 1)

        self.init_weights()

    def forward(self, input_ids, input_mask=None):
        # bert_out: # sequence_output, pooled_output, (hidden_states), (attentions)
        bert_out = self.bert(input_ids, input_mask=input_mask)
        # sequence_out.shape: (batch_size, seq_len, hidden_size), pooled_out.shape: (batch_size, hidden_size)
        sequence_out, pooled_out = bert_out[:2]

        logits = self.classifier(pooled_out)
        outputs = (logits,) + bert_out[2:]
        # logits, (hidden_states), (attentions)
        return outputs

class BertEpitopeClassificationTest(unittest.TestCase):
    def test_forward(self):
        model = BertEpitopeClassification.from_pretrained('../data/bert-base/', output_hidden_states=True, output_attentions=True)
        config = model.config
        tokenizer = TAPETokenizer(vocab='iupac')
        sequence = 'GCTVEDRCLIGMGAILLNGCVIGSGSLVAAGALITQ'
        token_ids = torch.tensor([tokenizer.encode(sequence)])
        input_len = len(token_ids[0])
        logits, hidden_states, attentions = model(token_ids)

        self.assertEqual((1, 1), logits.shape)
        self.assertEqual(config.num_hidden_layers + 1, len(hidden_states))
        expected_shape = (1, input_len, config.hidden_size)
        self.assertTrue(all(map(lambda x: expected_shape == x.shape, hidden_states)))

        self.assertEqual(config.num_hidden_layers, len(attentions))
        expected_shape = (1, config.num_attention_heads, input_len, input_len)
        self.assertTrue(all(map(lambda x: expected_shape == x.shape, attentions)))

if __name__ == '__main__':
    unittest.main()
