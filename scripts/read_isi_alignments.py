from amr_utils.alignments import AMR_Alignment
from amr_utils.amr_readers import AMR_Reader
import penman


from penman.model import Model
class TreePenmanModel(Model):
    def deinvert(self, triple):
        return triple

    def invert(self, triple):
        return triple

def node_map(amr1, amr2):
    node_labels = {}
    for n in amr1.nodes:
        candidate_nodes = [n2 for n2 in amr2.nodes if amr1.nodes[n].lower()==amr2.nodes[n2].lower()]

        if len(candidate_nodes)>1:
            neighbors = {n2: [f'{amr2.nodes[s]}_{r}_{amr2.nodes[t]}'.lower() for s, r, t in amr2.edges if n2 in [s, t]] for n2 in candidate_nodes}
            n_neighbors = [f'{amr1.nodes[s]}_{r}_{amr1.nodes[t]}'.lower() for s, r, t in amr1.edges if n in [s, t]]
            candidate_nodes = [n2 for n2 in candidate_nodes if set(neighbors[n2]) == set(n_neighbors)]
        if len(candidate_nodes)>1:
            neighbors2 = {n2: [s for s, r, t in amr2.edges if n2 == t] + [t for s, r, t in amr2.edges if n2 == s] for n2 in
                          candidate_nodes}
            neighbors2 = {n2: [f'{amr2.nodes[s]}_{r}_{amr2.nodes[t]}'.lower() for s, r, t in amr2.edges if s in neighbors2[n2] or t in neighbors2[n2]] for n2 in
                          candidate_nodes}
            n_neighbors2 = [s for s, r, t in amr1.edges if n == t] + [t for s, r, t in amr1.edges if n == s]
            n_neighbors2 = [f'{amr1.nodes[s]}_{r}_{amr1.nodes[t]}'.lower() for s, r, t in amr1.edges if s in n_neighbors2 or t in n_neighbors2]
            candidate_nodes = [n2 for n2 in candidate_nodes if set(neighbors2[n2]) == set(n_neighbors2)]
        if len(candidate_nodes)!=1:
            raise Exception('No match found:', amr1.id, n)
        node_labels[n] = candidate_nodes[0]
    return node_labels

def edge_map(amr1, amr2):
    node_labels = node_map(amr1, amr2)
    edge_labels = {}
    for s,r,t in amr1.edges:
        candidate_edges = [(s2,r2,t2) for s2,r2,t2 in amr2.edges if node_labels[s]==s2 and r.lower()==r2.lower() and node_labels[t]==t2]
        if len(candidate_edges)!=1:
            raise Exception('No match found:', amr1.id, s,r,t)
        edge_labels[(s,r,t)]  = candidate_edges[0]
    return edge_labels





def main():
    file = '../data/szubert/szubert_amrs.isi_alignments.txt'
    ids_file = '../data/szubert/szubert_ids.isi.txt'
    output = '../data/szubert/szubert_amrs.isi.txt'

    amr_file1 = '../data/ldc_train.txt'
    amr_file2 = '../data/szubert/szubert_amrs.txt'
    reader = AMR_Reader()
    amrs = reader.load(amr_file1, remove_wiki=True)
    szubert_amrs = reader.load(amr_file2, remove_wiki=True)
    szubert_amr_ids = [amr.id for amr in szubert_amrs]
    amrs += szubert_amrs
    amrs = {amr.id:amr for amr in amrs}

    amr_ids = []
    with open(ids_file, encoding='utf8') as f:
        for line in f:
            if line:
                amr_ids.append(line.strip())

    isi_amrs, isi_alignments = reader.load(file, output_alignments=True)

    subgraph_alignments = {}
    relation_alignments = {}
    for isi_amr in isi_amrs:
        if isi_amr.id not in szubert_amr_ids: continue
        amr = amrs[isi_amr.id]
        if len(amr.tokens)!=len(isi_amr.tokens):
            raise Exception('Inconsistent Tokenization:', amr.id)
        node_labels = node_map(isi_amr, amr)
        edge_labels = edge_map(isi_amr, amr)
        isi_aligns = isi_alignments[amr.id]
        subgraph_alignments[amr.id] = []
        relation_alignments[amr.id] = []
        for i,tok in enumerate(amr.tokens):
            aligns = [align for align in isi_aligns if i in align.tokens]
            nodes = [node_labels[n] for align in aligns for n in align.nodes]
            edges = [edge_labels[e] for align in aligns for e in align.edges]
            subgraph_alignments[amr.id].append(AMR_Alignment(type='subgraph', tokens=[i], nodes=nodes))
            relation_alignments[amr.id].append(AMR_Alignment(type='relation', tokens=[i], edges=edges))
    reader.save_alignments_to_json(output.replace('.txt','.subgraph_alignments.json'), subgraph_alignments)
    reader.save_alignments_to_json(output.replace('.txt', '.relation_alignments.json'), relation_alignments)

    for amr in szubert_amrs:
        if amr.id not in subgraph_alignments:
            raise Exception('Missing AMR:',amr.id)

if __name__=='__main__':
    main()