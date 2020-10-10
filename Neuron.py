import random
class Neuron():
    def __init__(self, ID, Neuron_map, tie_values, value = None):
        self.ID = ID
        self.childs = []
        if value == None:
            self.value = 0
        else:
            self.value = value
        self.tie_values = tie_values
        self.Neuron_map = Neuron_map
        self.Neuron_map[ID] = self
        self.parents = []
        self.activation_function = None
        self.d_activation_function = None
        self.error = 0
        self.non_derived_value = self.value

    def activation_function_applyer(self):
        if self.activation_function == None:
            return self.value
        return self.activation_function(self.value)

    def d_activation_function_applyer(self):
        if self.activation_function == None:
            return self.value
        return self.d_activation_function(self.non_derived_value)

    def tie(self, child_ID, value):
        self.childs.append(child_ID)
        child = self.get_child_by_ID(child_ID)
        self.tie_values[(child_ID, self.ID)] = value
        self.tie_values[(self.ID, child_ID)] = value
        child.parents.append(self.ID)

    def get_child_by_ID(self, Neuron_id):
        if Neuron_id not in self.Neuron_map:
            print("Error: A child is not connected!")
            exit()
            return None
        return self.Neuron_map[Neuron_id]

    def calculate_proportional_error_of_parents(self):
        total_error = 0
        for parent in self.parents:
            current_parrent = self.get_child_by_ID(parent)
            current_parrent.error += self.error * (self.tie_values[(current_parrent.ID, self.ID)])

    def feed_forward(self):
        for child_index in range(len(self.childs)):
            child_tenson = self.get_child_by_ID(self.childs[child_index])
            child_tenson.value += self.value * self.tie_values[(self.ID, child_tenson.ID)]
            child_tenson.non_derived_value = child_tenson.value

    def optimize(self, lr):
        self.optimize_sgd(lr)

    def optimize_sgd(self, lr):
         for parent in self.parents:
            current_parrent = self.get_child_by_ID(parent)
            current_gradient = self.tie_values[(current_parrent.ID, self.ID)] + self.error * lr * self.d_activation_function_applyer() * current_parrent.value
            self.tie_values[(current_parrent.ID, self.ID)] = current_gradient
            self.tie_values[(self.ID, current_parrent.ID)] = current_gradient

    def optimize_error(self, lr):
         for parent in self.parents:
            current_parrent = self.get_child_by_ID(parent)
            current_gradient = self.tie_values[(current_parrent.ID, self.ID)] + self.error * random.uniform(0, 0.2) * self.d_activation_function_applyer() * current_parrent.value
            self.tie_values[(current_parrent.ID, self.ID)] = current_gradient
            self.tie_values[(self.ID, current_parrent.ID)] = current_gradient