from amr_utils import display_alignments
from amr_utils.style import HTML_AMR


class Alignment_Display:

    @staticmethod
    def style(amrs, outfile, alignments):
        output = HTML_AMR.style(amrs[:5000],
                                assign_node_color=Alignment_Display.node_color,
                                assign_token_color=Alignment_Display.token_color,
                                assign_node_desc=Alignment_Display.node_desc,
                                assign_token_desc=Alignment_Display.token_desc,
                                assign_edge_desc=Alignment_Display.edge_desc,
                                assign_edge_color=Alignment_Display.edge_color,
                                other_args=alignments)

        with open(outfile, 'w+', encoding='utf8') as f:
            f.write(output)

    @staticmethod
    def node_desc(amr, n, alignments):
        align = amr.get_alignment(alignments, node_id=n)
        if align and align.type == 'subgraph':
            return ' '.join(amr.tokens[t] for t in align.tokens)
        if align and align.type.startswith('dupl'):
            return '<duplicate>'
        return ''

    @staticmethod
    def node_color(amr, n, alignments):
        align = amr.get_alignment(alignments, node_id=n)
        if align and align.type == 'subgraph':
            return 'green'
        if align:
            return 'blue'
        return ''

    @staticmethod
    def edge_desc(amr, e, alignments):
        align = amr.get_alignment(alignments, edge=e)
        if align and align.type == 'relation':
            return ' '.join(amr.tokens[t] for t in align.tokens)
        return ''

    @staticmethod
    def edge_color(amr, e, alignments):
        align = amr.get_alignment(alignments, edge=e)
        if align and align.nodes:
            return 'green'
        if align:
            return 'red'
        return ''

    @staticmethod
    def token_desc(amr, tok, alignments):
        align = amr.get_alignment(alignments, token_id=tok)
        if align and align.type.startswith('dupl'):
            return '<duplicate>'
        return display_alignments.get_token_aligned_subgraph(amr, tok, alignments)

    @staticmethod
    def token_color(amr, tok, alignments):
        align = amr.get_alignment(alignments, token_id=tok)
        if align and align.type == 'subgraph':
            return 'green'
        if align:
            return 'blue'
        return ''
