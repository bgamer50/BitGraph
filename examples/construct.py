import argparse
import sys, os
import re
import warnings

import numpy as np

import rmm
from rmm.allocators.torch import rmm_torch_allocator
from rmm.allocators.cupy import rmm_cupy_allocator

# Initialize shared allocator to prevent fragmentation
rmm.reinitialize(devices=0, pool_allocator=False, managed_memory=False)

import cupy
cupy.cuda.set_allocator(rmm_cupy_allocator)

import torch
torch.cuda.change_current_allocator(rmm_torch_allocator)

import cudf

sys.path.append('/mnt/bitgraph')
sys.path.append('/mnt/gremlin++')
from pybitgraph import BitGraph

from preprocess import Sentence_Transformer, Word2Vec_Transformer
from transformers import AutoModel, AutoTokenizer
torch.set_float32_matmul_precision('high')

def read_wiki_data(fname, skip_empty=True):
    df = cudf.read_json('/mnt/para_with_hyperlink.jsonl', lines=True)

    mentions = df.mentions.explode()
    mentions = mentions[~mentions.struct.field('sent_idx').isna()]
    mentions = mentions[~mentions.struct.field('ref_ids').isna()]

    df['sentence_offsets'] = cupy.concatenate([
        cupy.array([0]),
        df.sentences.list.len().cumsum().values[:-1]
    ])

    destinations_m = mentions.struct.field('ref_ids').list.get(0).astype('int64').values
    sources_m = mentions.struct.field('sent_idx').values + df.sentence_offsets[mentions.index].values + len(df)

    if skip_empty:
        # Does not add vertices/edges for vertices with no embedding
        f = destinations_m < len(df)
        destinations_m = destinations_m[f]
        sources_m = sources_m[f]
        del f

    eim = torch.stack([
        torch.as_tensor(sources_m, device='cuda'),
        torch.as_tensor(destinations_m, device='cuda'),
    ])

    sentences = df.sentences.explode().reset_index().rename({"index": 'article'},axis=1)

    sources_s = sentences.index.values + len(df)
    destinations_s = sentences.article.values
    eis = torch.stack([
        torch.as_tensor(sources_s, device='cuda'),
        torch.as_tensor(destinations_s, device='cuda'),
    ])

    eix = torch.concatenate([eim,eis],axis=1)
    del eis
    del eim

    return eix, df.title.to_pandas(), sentences.sentences.to_pandas()


def read_embeddings(graph, directory, td):
    ex = re.compile(r'part_([0-9]+)\_([0-9]+).pt')
    def fname_to_key(s):
        m = ex.match(s)
        return int(m[1]), int(m[2])

    ix = 0

    for emb_type in ['titles', 'sentences']:
        path = os.path.join(directory, emb_type)
        files = os.listdir(path)

        files = sorted(files, key=fname_to_key)
        for f in files:
            e = torch.load(os.path.join(path, f), weights_only=True, map_location='cuda').reshape((-1, td))

            print(ix, e.shape)
            graph.set_vertex_embeddings('emb', ix, ix + e.shape[0] - 1, e)
            
            ix += e.shape[0]
            del e


def getem_roberta(model, tokenizer, text):
    t = tokenizer(text, return_tensors='pt')
    while t.input_ids.shape[1] > 512:
        a = a[:-10]
        t = tokenizer(a, return_tensors='pt')
    return model(t.input_ids, t.attention_mask)


def getem_w2v(model, text):
    return model(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip_empty_vertices', type=bool, required=False, default=True)
    parser.add_argument('--property_storage', type=str, required=False, default='managed')
    parser.add_argument('--fname', type=str, required=True)
    parser.add_argument('--embeddings_dir', type=str, required=True)
    parser.add_argument('--embedding_type', type=str, required=True)
    parser.add_argument('--search_query', type=str, required=True)
    parser.add_argument('--w2v_path', required=False, type=str, default='./GoogleNews-vectors-negative300.bin.gz')

    args = parser.parse_args()

    eix, titles, sentences = read_wiki_data(
        args.fname,
        args.skip_empty_vertices
    )
    print('read wiki data')

    graph = BitGraph(
        'int64',
        'int64',
        'DEVICE',
        'DEVICE',
        args.property_storage.upper(),
    )

    graph.add_vertices(eix.max() + 1)
    graph.add_edges(eix[0], eix[1], 'link')

    read_embeddings(
        graph,
        args.embeddings_dir,
        td=300 if args.embedding_type == 'w2v' else 1024,
    )    
    print('read embeddings into graph')
    
    g = graph.traversal()
    print('constructed graph')

    if args.embedding_type == 'w2v':
        import gensim
        warnings.warn("Word2Vec encoder is for testing/debugging purposes only!")
        module = Word2Vec_Transformer(
            gensim.models.KeyedVectors.load_word2vec_format(args.w2v_path, binary=True),
            dim=300,
        )
        getem = lambda t : getem_w2v(module, t)
    elif args.embedding_type == 'roberta':
        model = AutoModel.from_pretrained('sentence-transformers/all-roberta-large-v1')
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-roberta-large-v1')
        
        mod = Sentence_Transformer(model).cuda()
        import torch._dynamo
        torch._dynamo.reset()

        module = torch.compile(mod, fullgraph=True)
        getem = lambda t : getem_roberta(module, tokenizer, t)
    else:
        raise ValueError("Expected 'w2v' or 'roberta' for embedding type")

    qe = getem(args.search_query)
    vids = g.V().like('emb', [qe], 4).toArray()

    f = vids < len(titles)
    article_ids = vids[f]
    sentence_ids = vids[~f] - len(titles)

    print('articles:', titles.iloc[article_ids])
    print('sentences:', sentences.iloc[sentence_ids])
