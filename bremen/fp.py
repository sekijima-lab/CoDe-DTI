"""Experimental neural fingerprint helper on the maintained CPU backend."""
import torch
from torch import nn
from .autoencoder import DL
AtomIdMax = 100

class nfp:
    def __init__(self, d, f, R):
        self.d, self.f, self.R = d, f, R
        self.model = nn.Module()
        self.model.g = nn.ModuleList(DL(0, 1, f) for _ in range(AtomIdMax))
        self.model.H = nn.ModuleList(nn.ModuleList(DL(0, f, f) for _ in range(R)) for _ in range(5))
        self.model.W = nn.ModuleList(DL(0, f, d) for _ in range(R))
        self.optimizer = torch.optim.Adam(self.model.parameters())
    def get_nfp(self, mol, train=True):
        with torch.set_grad_enabled(train):
            atoms = list(mol.GetAtoms())
            r = [self.model.g[atom.GetAtomicNum()](torch.ones(1, 1), test=True) for atom in atoms]
            fp = torch.zeros(1, self.d)
            for layer in range(self.R):
                values = []
                for atom in atoms:
                    v = r[atom.GetIdx()]
                    for neighbour in atom.GetNeighbors():
                        v = v + r[neighbour.GetIdx()]
                    values.append(torch.tanh(self.model.H[min(5, max(1, atom.GetDegree()))-1][layer](v, test=True)))
                r = values
                for value in r:
                    fp = fp + torch.softmax(self.model.W[layer](value, test=True), dim=1)
            return fp
