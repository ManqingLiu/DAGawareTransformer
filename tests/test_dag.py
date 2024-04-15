import unittest
import torch

from src.models.DAG_aware_transformer import DAGTransformer

class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.batch_size = 8
        self.num_t_categories = 2
        self.num_l_categories = 20
        self.num_y_categories = 15
        t = torch.randint(0, self.num_t_categories, (self.batch_size, 1))
        l = torch.randint(0, self.num_l_categories, (self.batch_size, 1))
        y = torch.randint(0, self.num_y_categories, (self.batch_size, 1))
        x = {'t': t, 'l': l, 'y': y}

        nodes = ['t', 'l', 'y']
        input_nodes = {'t': {'num_categories': self.num_t_categories},
                       'l': {'num_categories': self.num_l_categories},
                       'y': {'num_categories': self.num_y_categories}}

        output_nodes = {'t': {'num_categories': self.num_t_categories},
                        'y': {'num_categories': self.num_y_categories}}

        num_nodes = len(nodes)
        self.dag = {}

        edges = {}
        edges['t'] = ['y']
        edges['l'] = ['t', 'y', 'l']
        node_ids = dict(zip(nodes, range(num_nodes)))

        self.dag['edges'] = edges
        self.dag['node_ids'] = dict(node_ids)
        self.dag['input_nodes'] = input_nodes
        self.dag['output_nodes'] = output_nodes

        self.x = {'t': t, 'l': l, 'y': y}
    def test_forward_pass_shape(self):
        # Make some data

        model = DAGTransformer(dag=self.dag)
        outputs = model(self.x)
        y_hat = outputs['y']
        t_hat = outputs['t']
        # Check that the output is the correct shape
        assert y_hat.shape == (self.batch_size, 15) and t_hat.shape == (self.batch_size, 2)

    def test_loss_function(self):
        model = DAGTransformer(dag=self.dag)
        outputs = model(self.x)

        # TODO: Move this to its own function
        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        losses = []
        for output_name in outputs.keys():
            output = outputs[output_name]
            labels = self.x[output_name].squeeze()
            losses.append(loss_fn(output, labels))

        total_loss = sum(losses) / len(losses)
        total_loss.backward()
        self.assertFalse(torch.isnan(total_loss))

    def test_one_step(self):
        model = DAGTransformer(dag=self.dag)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)

        outputs = model(self.x)

        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        losses = []
        for output_name in outputs.keys():
            output = outputs[output_name]
            labels = self.x[output_name].squeeze()
            losses.append(loss_fn(output, labels))

        total_loss = sum(losses) / len(losses)
        total_loss.backward()
        opt.step()
        opt.zero_grad()
        initial_loss = total_loss.item()

        outputs = model(self.x)
        losses = []
        for output_name in outputs.keys():
            output = outputs[output_name]
            labels = self.x[output_name].squeeze()
            losses.append(loss_fn(output, labels))

        total_loss = sum(losses) / len(losses)

        assert total_loss.item() < initial_loss

    def test_full_backwardpass(self):
        model = DAGTransformer(dag=self.dag)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)

        for i in range(100):
            outputs = model(self.x)

            loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
            losses = []
            for output_name in outputs.keys():
                output = outputs[output_name]
                labels = self.x[output_name].squeeze()
                losses.append(loss_fn(output, labels))

            total_loss = sum(losses) / len(losses)
            total_loss.backward()
            opt.step()
            opt.zero_grad()

        assert total_loss.item() < 0.1


if __name__ == '__main__':
    unittest.main()
