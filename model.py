import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DeepQNetwork(nn.Module):
    def __init__(self,  input_dims, n_actions, seed, lr, fc1_dims=64, fc2_dims=64):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.lr = lr
        
        self.seed = T.manual_seed(seed)
        
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions
    
class DuelingDeepQNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, seed, lr, fc1_dims=64, fc2_dims=64):
        super(DuelingDeepQNetwork, self).__init__()
        
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.lr = lr
        
        self.seed = T.manual_seed(seed)
        
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        
        # V function
        self.V = nn.Linear(self.fc2_dims, 1)
        # A function
        self.A = nn.Linear(self.fc2_dims, self.n_actions)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        V = self.V(x)
        A = self.A(x)
        # We construct Q from V and A
        # V is scalar that shifts A by a scalar quantity
        # Subtract of Mean of Advantage
        Q = V + (A - A.mean(dim=1, keepdim=True))

        return Q