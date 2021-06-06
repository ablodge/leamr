from amr_utils.amr_readers import AMR_Reader


def main():
    file = 'data/szubert/szubert_amrs.txt'
    output = 'data/szubert/szubert_amrs.jamr.txt'

    reader = AMR_Reader()
    amrs = reader.load(file, remove_wiki=True)

    with open(output, 'w+') as f:
        for amr in amrs:
            f.write('# ::id '+amr.id+'\n')
            tokens = [t for t in amr.tokens]
            for i,t in enumerate(tokens):
                if t[0]=='@' and t[-1]=='@' and len(t)==3:
                    tokens[i] = t[1]
            f.write('# ::snt ' + ' '.join(tokens) + '\n')
            graph_string = amr.graph_string().replace('/',' / ')
            f.write(graph_string)
            

if __name__=='__main__':
    main()