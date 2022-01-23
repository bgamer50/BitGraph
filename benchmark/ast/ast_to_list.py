from sys import argv
import networkx as nx
import re
import unicodedata

expr = re.compile(r'[|\-\s]+([A-Za-z]+)\s+')
esc = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]')

graph = nx.Graph()

filename_in = argv[1]
filename_out_nodes = argv[2]
filename_out_edges = argv[3]

stack = ['root']
with open(filename_in, 'r') as infile:
    with open(filename_out_nodes, 'w') as outfile_nodes:
        outfile_nodes.write('NAME,INFO,LEVEL\n')
        outfile_nodes.write('root,root,0\n')

        with open(filename_out_edges, 'w') as outfile_edges:
            for i, line in enumerate(infile.readlines()):
                if i % 1000 == 0:
                    print(i)

                line = esc.sub('', line.strip())
                if '`' not in line:
                    #line = ''.join([c for c in line if unicodedata.category(c)[0] != 'C'])

                    level = len([c for c in line if c == '|']) + 1
                    
                    if level > 1:
                        m = expr.match(line)
                        if m is not None:
                            info_str = m.groups()[0].replace(' ','_')
                        else:
                            info_str = f'ent_ln{i}'
                    else:
                        info_str = 'head'

                    
                    #graph.add_node(i, info=info_str, level=level)
                    outfile_nodes.write(f'{i},{info_str},{level}\n')
                    stack = stack[:level]

                    if len(stack) > 0:
                        #graph.add_edge(i, stack[-1])
                        outfile_edges.write(f'{i} {stack[-1]}\n')

                    stack.append(i)

#nx.write_graphml(graph, filename_out)