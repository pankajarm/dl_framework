# import usual suspects

import numpy as np

class InvalidInputException(Exception):
    def __init__(self):
        print("Input nodes and Weights are not of the same length")


class InconsistentArrayLengthException(Exception):
    def __init__(self):
        print("Array length of given inputs do not match")

class Node(object):
    """
    Base class for nodes in the network.
    Arguments:
        `inbound_nodes`: A list of nodes with edges into this node.
    """
    def __init__(self, inbound_nodes=[]):
         # Node(s) from which this Node receives values
        self.inbound_nodes = inbound_nodes

        # Node(s) to which this Node passes values
        self.outbound_nodes = []

        # Attribute for caching the gradient values
        self.gradients = {}

        # A calculated value for the node
        self.value = None

        # For each inbound Node here, add this Node as an outbound Node to _that_ Node.
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)


    def forward(self):
        """ Forward propagation.
            Compute the output value based on `inbound_nodes` and
            store the result in self.value.
        """
        raise NotImplementedError


    def backward(self):
        """ Back propagation.
            Compute the output value based on `outbound_nodes` and
            store the result in self.value.
        """
        raise NotImplementedError

class Input(Node):
    """ Input node has no inbound nodes, hence no need to specify inbound
        nodes parameter while initializing
    """

    def __init__(self):
        Node.__init__(self)

        """ NOTE: Input node is the only node where the value
            may be passed as an argument to forward().
            All other node implementations should get the value
            of the previous node from self.inbound_nodes
        """

    def forward(self):
        # Do nothing because nothing is calculated.
        pass

    def backward(self):
        # An Input node has no inputs so the gradient (derivative)
        # is zero.
        # The key, `self`, is reference to this object.
        self.gradients = {self: 0}

        # Weights and bias may be inputs, so you need to sum
        # the gradient from output gradients.
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            self.gradients[self] += grad_cost * 1

# Linear Activation Node
class Linear(Node):
    def __init__(self, X, W, b):
        Node.__init__(self, [X, W, b])

    def forward(self):
        # Fetching input nodes, weights and bias
        X = self.inbound_nodes[0].value
        W = self.inbound_nodes[1].value
        b = self.inbound_nodes[2].value

        self.value = np.dot(X, W) + b

    def backward(self):
        # Initialize a partial for each of the inbound_nodes i.e. X, W, b
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}

        # Cycle through the outputs. The gradient will change depending
        # on each output, so the gradients are summed over all outputs.
        for n in self.outbound_nodes:
            # Get the partial of the cost with respect to this node.
            grad_cost = n.gradients[self]

            # Set the partial of the loss with respect to this node's inputs i.e X.
            # Partial derivative w.r.t. X is W and for ease of matrix multiplication transpose of W is used
            self.gradients[self.inbound_nodes[0]] += np.dot(grad_cost, self.inbound_nodes[1].value.T)

            # Set the partial of the loss with respect to this node's weights i.e. W.
            # Partial derivative w.r.t. W is X and for ease of matrix multiplication transpose of X is used
            self.gradients[self.inbound_nodes[1]] += np.dot(self.inbound_nodes[0].value.T, grad_cost)

            # Set the partial of the loss with respect to this node's bias i.e. b.
            # Partial derivative w.r.t. b is 1 hence summing up all the grad_cost across 0th axis
            self.gradients[self.inbound_nodes[2]] += np.sum(grad_cost, axis=0, keepdims=False)

# Sigmoid Activation Node
class Sigmoid(Node):
    def __init__(self, X):
        Node.__init__(self, [X])

    def _sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def forward(self):
        X = self.inbound_nodes[0].value
        self.value = self._sigmoid(X)

    def backward(self):
        """
        Calculates the gradient using the derivative of
        the sigmoid function.
        """
        # Initialize the gradients to 0.
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}

        # Cycle through the outputs. The gradient will change depending
        # on each output, so the gradients are summed over all outputs.
        for n in self.outbound_nodes:
            # Get the partial of the cost with respect to this node.
            grad_cost = n.gradients[self]

            self.gradients[self.inbound_nodes[0]] += grad_cost * self.value * (1 - self.value)

# Mean Square Error Calculation Node
class MSE(Node):
    def __init__(self, y, a):
        Node.__init__(self, [y, a])

    def forward(self):
        # Converting to a row matrix
        y = self.inbound_nodes[0].value.reshape(-1, 1)
        a = self.inbound_nodes[1].value.reshape(-1, 1)

        if len(y) == len(a):
            m = len(y)

            error = 0

            for i in range(0, m):
                error += np.square(y[i] - a[i])

            error /= m

            self.value = error
        else:
            raise InconsistentArrayLengthException

    def backward(self):
        """
        Calculates the gradient of the cost.
        This is the final node of the network so outbound nodes
        are not a concern.
        """
        # Converting to a row matrix
        y = self.inbound_nodes[0].value.reshape(-1, 1)
        a = self.inbound_nodes[1].value.reshape(-1, 1)

        if len(y) == len(a):
            m = len(y)

            self.gradients[self.inbound_nodes[0]] = (2 / m) * (y - a)
            self.gradients[self.inbound_nodes[1]] = (-2 / m) * (y - a)

        else:
            raise InconsistentArrayLengthException

def topological_sort(feed_dict):
    """
    Sort the nodes in topological order using Kahn's Algorithm.
    `feed_dict`: A dictionary where the key is a `Input` Node and the value is the respective value feed to that Node.
    Returns a list of sorted nodes.
    """

    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L

# Function to perform a forward pass
def forward_pass(sorted_nodes):
    """
    Performs a forward pass through a list of sorted nodes.
    Arguments:
        `output_node`: The output node of the graph (no outgoing edges).
        `sorted_nodes`: a topologically sorted list of nodes.
    Returns the output node's value
    """

    for n in sorted_nodes:
        n.forward()

# Function to perform a backward pass (Back Propogation)
def backward_pass(sorted_nodes):
    """
    Performs a forward pass and a backward pass through a list of sorted Nodes.
    Arguments:
        `graph`: The result of calling `topological_sort`.
    """
    # Forward pass
    forward_pass(sorted_nodes)

    # Backward pass
    # see: https://docs.python.org/2.3/whatsnew/section-slices.html
    for n in sorted_nodes[::-1]:
        n.backward()

# Stochastic Gradient Descent Update
def sgd_update(trainables, learning_rate = 0.01):

    # Updating the trainable values as gradient descent equation
    for t in trainables:
       t.value = t.value - learning_rate * t.gradients[t]
