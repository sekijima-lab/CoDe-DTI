from rdkit import Chem

class SmilesRepository(list):
    def __init__(self, smiles_list_file, sanitize=True):
        super().__init__()
        with open(smiles_list_file) as fp:
            for line in fp:
                drugbank_id, smiles = tuple(line.strip().split(' '))
                super().append({
                    'drugbank_id': drugbank_id,
                    'mol': Chem.MolFromSmiles(smiles, sanitize=sanitize),
                    'smiles': smiles})

    def append(self, p_object):
        raise NotImplementedError()