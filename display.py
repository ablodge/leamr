from amr_utils import display_alignments
from amr_utils.style import HTML_AMR


class Display:

    @staticmethod
    def style(amrs, outfile):
        output = HTML_AMR.style(amrs[:5000],
                                assign_node_color=Display.node_color,
                                assign_token_color=Display.token_color,
                                assign_node_desc=Display.node_desc,
                                assign_token_desc=Display.token_desc,
                                assign_edge_desc=Display.edge_desc,
                                assign_edge_color=Display.edge_color)

        with open(outfile, 'w+', encoding='utf8') as f:
            f.write(output)

    @staticmethod
    def node_desc(amr, n):
        align = amr.get_alignment(node_id=n)
        if align and align.type == 'subgraph':
            return ' '.join(amr.tokens[t] for t in align.tokens)
        if align and align.type.startswith('dupl'):
            return '<duplicate>'
        return ''

    @staticmethod
    def node_color(amr, n):
        align = amr.get_alignment(node_id=n)
        if align and align.type == 'subgraph':
            return 'green'
        if align:
            return 'blue'
        return ''

    @staticmethod
    def edge_desc(amr, e):
        align = amr.get_alignment(edge=e)
        if align and align.type == 'relation':
            return ' '.join(amr.tokens[t] for t in align.tokens)
        return ''

    @staticmethod
    def edge_color(amr, e):
        align = amr.get_alignment(edge=e)
        if align and align.type == 'relation':
            if not align.nodes:
                return 'red'
            return 'grey'
        if align:
            return 'grey'
        return ''

    @staticmethod
    def token_desc(amr, tok):
        align = amr.get_alignment(token_id=tok)
        if align and align.type.startswith('dupl'):
            return '<duplicate>'
        return display_alignments.get_token_aligned_subgraph(amr, tok)

    @staticmethod
    def token_color(amr, tok):
        align = amr.get_alignment(token_id=tok)
        if align and align.type == 'subgraph':
            return 'green'
        if align:
            return 'blue'
        return ''
