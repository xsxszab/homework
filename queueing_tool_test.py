
import numpy as np
import networkx as nx
import queueing_tool as qt

adj_list = {0: [1, 2, 3], 1: [4], 2: [5], 3: [6]}
#  queue structure
#   / --> 1 --> 4
#   0 --> 2 --> 5
#   \ --> 3 --> 6

edge_types = {0: {1: 1, 2: 1, 3: 1},  # 0->1, 0->2, 0->3 : type 1
              1: {4: 2}, 2: {5: 2}, 3: {6: 2}}  # 1->4, 2->5, 3->6: type 2


g = qt.adjacency2graph(adj_list, edge_type=edge_types)


def rate(t):
    return 25 + 350 * np.sin(np.pi * t / 2) ** 2  # customer arrival rate


def arr_f(t):
    return qt.poisson_random_measure(t, rate, 375)  # rate is the expectation of poisson stream


def ser_f(t):
    return t + np.random.exponential(0.2 / 2.1)  # service rate


classes = {1: qt.QueueServer, 2: qt.QueueServer}
queue_args = {
    1: {
        'arrival_f': arr_f,
        'service_f': lambda t: t + 0.001,
        'AgentFactory': qt.GreedyAgent
    },

    2: {
        'num_servers': 1,
        'service_f': ser_f
    }
}

qn = qt.QueueNetwork(g=g, q_classes=classes, q_args=queue_args)
qn.initialize(edge_type=1)  # indicate that customers come from edge type 1

qn.start_collecting_data()
qn.simulate(t=1.9)  # simulate 1.9 units of time
data = qn.get_queue_data()  # get simulated data, 6 cols see docs for details
