import argparse
import sys, os
import re
import warnings
import json

import numpy as np

from time import perf_counter

from math import cos, pi

from sklearn.model_selection import ParameterSampler

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
import pandas

from torch_geometric.data import Data
from torch_geometric.nn import GRetriever, GAT
from torch_geometric.nn.nlp import LLM

from torch.nn.utils import clip_grad_norm_

sys.path.append('/mnt/bitgraph')
sys.path.append('/mnt/gremlin++')
from pybitgraph import BitGraph
from pygremlinxx import GraphTraversal
__ = lambda : GraphTraversal()

from preprocess import Sentence_Transformer, Word2Vec_Transformer
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline
)

torch.set_float32_matmul_precision('high')

def read_wiki_data(fname, skip_empty=True):
    df = cudf.read_json(fname, lines=True)

    mentions = df.mentions.explode()
    mentions = mentions[~mentions.struct.field('sent_idx').isna()]
    mentions = mentions[~mentions.struct.field('ref_ids').isna()]

    slens = df.sentences.list.len().astype('int64')
    slens[(slens==0)] = 1

    df['sentence_offsets'] = cupy.concatenate([
        cupy.array([0]),
        slens.cumsum().values[:-1]
    ])

    mix = torch.as_tensor(
        mentions.struct.field('ref_ids').list.get(0).astype('int64').values,
        device='cuda'
    )
    ids = torch.as_tensor(df.id.astype('int64').values, device='cuda')
    vals, inds = torch.sort(ids)
    

    destinations_m = inds[torch.searchsorted(vals, mix)]
    sources_m = torch.as_tensor(
        mentions.struct.field('sent_idx').values + df.sentence_offsets[mentions.index].values + len(df),
        device='cuda'
    )

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

    destinations_s = sentences.index.values + len(df)
    sources_s = sentences.article.values
    eis = torch.stack([
        torch.as_tensor(sources_s, device='cuda'),
        torch.as_tensor(destinations_s, device='cuda'),
    ])

    eix = torch.concatenate([eim,eis],axis=1)
    del eis
    del eim

    return eix, df.title.to_pandas(), sentences.to_pandas()


def read_embeddings(graph, directory, td, map_location='cuda'):
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
            e = torch.load(os.path.join(path, f), weights_only=True, map_location=map_location).reshape((-1, td))
            if map_location == 'cpu':
                e = e.pin_memory()

            print(ix, e.shape)
            graph.set_vertex_embeddings('emb', ix, ix + e.shape[0] - 1, e)
            
            ix += e.shape[0]
            del e

    graph.make_vertex_embedding_index('emb')

def getem_roberta(model, tokenizer, text):
    with torch.no_grad():
        t = tokenizer(text, return_tensors='pt')
        while t.input_ids.shape[1] > 512:
            a = a[:-10]
            t = tokenizer(a, return_tensors='pt')
        return model(t.input_ids.cuda(), t.attention_mask.cuda())


def getem_w2v(model, text):
    with torch.no_grad():
        return model(text)


def extract(entsList):
    words = []
    for ents in entsList:
        row = []
        for ent in ents:
            row.append(ent['word'])
        words.append(row)
    return words


def get_question_vertices(g, f, ner, question, e_limit, q_limit):
    start_time = perf_counter()
    ents = ner(question)
    emb_q = f(question)
    ent_embs = [f(ent['word']) for ent in ents]
    end_time = perf_counter()

    print('ner time:', end_time - start_time)

    ent_vids = [
        g.V().like('emb', [ent_emb], e_limit).toArray()
        for ent_emb in ent_embs if ent_emb.sum() != 0
    ]
    if len(ent_vids) == 0:
        q_limit = max(q_limit, 1)

    vids = cupy.concatenate(
        ent_vids + [
            g.V().like('emb', [emb_q], q_limit).toArray()
        ]
    )
    vids = torch.as_tensor(vids).cuda()
    vids = cupy.asarray(vids)

    print(ents)
    return vids, emb_q

def tune_query(g, f, ner, questions, contexts, grid, out_fname):
    with open(out_fname, 'a') as qf:
        for d in ParameterSampler(**grid):
            print('params:', d)
            start_time = perf_counter()
            total_err = 0.0
            total_size = 0
            for i in range(len(questions)):
                question = questions.iloc[i]
                context = contexts.iloc[i]

                print(f'question {i}: {question}')
            
                start_time_match = perf_counter()
                vids_q, emb_q = get_question_vertices(
                    g,
                    f,
                    ner,
                    question,
                    d['entity_vertex_match_limit'],
                    d['question_vertex_match_limit']
                )
                end_time_match = perf_counter()
                print('match time:', end_time_match - start_time_match)

                start_time_query = perf_counter()
                v_emb = g.V(vids_q)._as('s')._union([
                    __().out().order().by(__().similarity('emb', [emb_q])).limit(d['hop_0_outgoing_limit'])._as('h0'),
                    __()._in().order().by(__().similarity('emb', [emb_q])).limit(d['hop_0_incoming_limit'])._as('h0'),
                ])._union([
                    __().out().order().by(__().similarity('emb', [emb_q])).limit(d['hop_1_outgoing_limit'])._as('h1'),
                    __()._in().order().by(__().similarity('emb', [emb_q])).limit(d['hop_1_incoming_limit'])._as('h1'),
                ])._union([__().select('h0'), __().select('h1'), __().select('s')]).dedup().encode('emb').toArray().reshape((-1, emb_q.numel()))
                end_time_query = perf_counter()
                print('query time:', end_time_query - start_time_query)

                start_time_cmp = perf_counter()
                v_emb = torch.as_tensor(v_emb, device='cpu').detach()
                num_vertices = v_emb.shape[0]
                v_emb = v_emb.sum(0) / v_emb.shape[0]
                
                c_emb = torch.concat([f(c[0]) for c in context]).cpu()
                c_emb = c_emb.sum(0) / c_emb.shape[0]

                err = ((v_emb - c_emb)**2).sum()**0.5
                total_err += err
                total_size += num_vertices
                end_time_cmp = perf_counter()
                print('compare time:', end_time_cmp - start_time_cmp)
                print('error:', err)
                print('# vertices:', num_vertices)
            
            end_time = perf_counter()
            print('total err:', total_err)
            j = json.dumps({
                'params': d,
                'avg_error' : float(total_err / len(questions)),
                'avg_time': float((end_time - start_time) / len(questions)),
                'avg_vertices': float(total_size / len(questions))
            })
            qf.write(j +'\n')
            qf.flush()


def adjust_learning_rate(param_group, LR, epoch, num_epochs):
    # Decay the learning rate with half-cycle cosine after warmup
    # Adapted from the PyG G-Retriever Implementation
    # (credit: PyG team, Rishi Puri)
    min_lr = 5e-6
    warmup_epochs = 1
    if epoch < warmup_epochs:
        lr = LR
    else:
        lr = min_lr + (LR - min_lr) * 0.5 * (
            1.0 + cos(pi * (epoch - warmup_epochs) /
                            (num_epochs - warmup_epochs)))
    param_group['lr'] = lr
    return lr


def save_params_dict(model, save_path):
    # Adapted from the PyG G-Retriever Implementation
    # (credit: PyG team, Rishi Puri)
    state_dict = model.state_dict()
    param_grad_dict = {
        k: v.requires_grad
        for (k, v) in model.named_parameters()
    }
    for k in list(state_dict.keys()):
        if k in param_grad_dict.keys() and not param_grad_dict[k]:
            del state_dict[k]  # Delete parameters that do not require gradient
    torch.save(state_dict, save_path)


def load_params_dict(model, save_path):
    # Adapted from the PyG G-Retriever Implementation
    # (credit: PyG team, Rishi Puri)
    state_dict = torch.load(save_path)
    model.load_state_dict(state_dict)
    return model


def get_optimizer(model, lr):
    params = [p for _, p in model.named_parameters() if p.requires_grad]

    # This configuration is adapted from the PyG G-Retriever Implementation
    # (credit: PyG team, Rishi Puri)
    optimizer = torch.optim.AdamW([
        {
            'params': params,
            'lr': lr,
            'weight_decay': 0.05
        },
    ], betas=(0.9, 0.95))
    return optimizer


def coo_to_data(g, coo):
    data = Data()
    
    data.edge_index = torch.stack([
        torch.as_tensor(coo['dst'].astype('int64'), device='cuda'),
        torch.as_tensor(coo['src'].astype('int64'), device='cuda'),
    ])
    
    x = torch.as_tensor(
        g.V(coo['vid']).encode('emb').toArray(),
        device='cuda'
    )
    
    td = x.numel() // len(coo['vid'])
    print('td:', td)
    data.x = x.reshape((-1, td))
    
    data.n_id = torch.as_tensor(
        coo['vid'],
        device='cuda'
    )
    
    data.batch = torch.zeros((data.x.shape[0],), dtype=torch.int64, device='cuda')

    return data


def get_prompt(ner, g, f, qp, titles, sentences, question, rag_type='direct'):
    if rag_type in ['direct', 'combined']:
        vids_q, emb_q = get_question_vertices(
            g,
            f,
            ner,
            question,
            qp['entity_vertex_match_limit'],
            qp['question_vertex_match_limit']
        )

    if rag_type == 'direct':
        vids = g.V(vids_q)._as('s')._union([
            __().out().order().by(__().similarity('emb', [emb_q])).limit(qp['hop_0_outgoing_limit'])._as('h0'),
            __()._in().order().by(__().similarity('emb', [emb_q])).limit(qp['hop_0_incoming_limit'])._as('h0'),
        ])._union([
            __().out().order().by(__().similarity('emb', [emb_q])).limit(qp['hop_1_outgoing_limit'])._as('h1'),
            __()._in().order().by(__().similarity('emb', [emb_q])).limit(qp['hop_1_incoming_limit'])._as('h1'),
        ])._union([__().select('h0'), __().select('h1'), __().select('s')]).dedup().toArray()

        fx = (vids < len(titles))
        ix = vids[~fx].get() - len(titles)

        s = {
            titles.iloc[k]: (' '.join(sentences.sentences[v].tolist()))
            for k, v in sentences.iloc[ix].groupby('article').groups.items()
        }

        context = '\n'.join([f'{t} - {p}' for t, p in s.items()])
        prompt = f'Question: Given the information below, {question}\n{context}\nAnswer:'
        return (prompt, None)
    elif rag_type == 'combined':
        eids = g.V(vids_q)._union([
            __().outE().order().by(__().inV().similarity('emb', [emb_q])).limit(qp['hop_0_outgoing_limit'])._as('h0').inV(),
            __().inE().order().by(__().outV().similarity('emb', [emb_q])).limit(qp['hop_0_incoming_limit'])._as('h0').outV(),
        ])._union([
            __().outE().order().by(__().inV().similarity('emb', [emb_q])).limit(qp['hop_1_outgoing_limit'])._as('h1').inV(),
            __().inE().order().by(__().outV().similarity('emb', [emb_q])).limit(qp['hop_1_incoming_limit'])._as('h1').outV(),
        ])._union([__().select('h0'), __().select('h1')]).dedup().toArray()
        
        out = graph.subgraph_coo(eids)
        data = coo_to_data(g, out)
        
        prompt = f'Question: {question}\nAnswer:'
        return (prompt, data)
    elif rag_type == 'none':
        prompt = f'Question: {question}\nAnswer:'
        return (prompt, None)
    else:
        raise ValueError("Invalid rag type")


def test(model, ner, g, f, qp, titles, sentences, questions, answers, epochs=1, lr=1e-5, rag_type='direct'):
    model.eval()
    with torch.no_grad():
        for i in range(len(questions)):            
            question = questions.iloc[i]
            answer = answers.iloc[i]
            
            prompt, data = get_prompt(ner, g, f, qp, titles, sentences, question, rag_type=rag_type)
            print(f'Prompt {i}: {prompt} {answer}')

            if rag_type in ['direct', 'none']:
                loss = model(
                    [prompt],
                    [answer],
                    None,
                )
            else:
                loss = model(
                    question=[prompt],
                    x=data.x,
                    edge_index=data.edge_index,
                    batch=data.batch,
                    label=[answer],
                    edge_attr=None,
                    additional_text_context=None,
                )

            print(f'loss {i}: {loss}')
            total_loss = total_loss + float(loss)

        test_loss = total_loss / len(questions)
        print(f'Test Loss: {test_loss:4f}')


def train(model, ner, g, f, qp, titles, sentences, questions, answers, epochs=1, lr=1e-5, rag_type='direct'):
    optimizer = get_optimizer(model, lr)
    grad_steps = 2

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for i in range(len(questions)):
            optimizer.zero_grad()
            
            question = questions.iloc[i]
            answer = answers.iloc[i]

            prompt, data = get_prompt(ner, g, f, qp, titles, sentences, question, rag_type)
            print(f'Prompt {i}: {prompt} {answer}')

            if rag_type in ['direct', 'none']:
                loss = model(
                    [prompt],
                    [answer],
                    None,
                )
            else:
                loss = model(
                    question=[prompt],
                    x=data.x,
                    edge_index=data.edge_index,
                    batch=data.batch,
                    label=[answer],
                    edge_attr=None,
                    additional_text_context=None,
                )

            print(f'loss {i}: {loss}')
            loss.backward()
            
            clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)

            if (i + 1) % grad_steps == 0:
                adjust_learning_rate(
                    optimizer.param_groups[0],
                    lr,
                    i / len(questions) + epoch,
                    epochs
                )

            optimizer.step()
            epoch_loss = epoch_loss + float(loss)

            if (i + 1) % grad_steps == 0:
                lr = optimizer.param_groups[0]['lr']
        train_loss = epoch_loss / len(questions)
        print(f'Epoch {epoch}, Train Loss: {train_loss:4f}')

        print('Saving model...')
        fname = 'llm_direct_tuned.pt' if rag_type == 'direct' else 'gnn_llm_tuned.pt'
        save_params_dict(model, fname)
        print('Model saved successfully.')


def tune_llm(llm, questions, contexts, answers, epochs=1, lr=1e-5):
    optimizer = get_optimizer(llm, lr)
    grad_steps = 2

    for epoch in range(epochs):
        llm.train()
        epoch_loss = 0.0
        for i in range(len(questions)):
            optimizer.zero_grad()
            
            question = questions.iloc[i]
            context = None if contexts is None else contexts.iloc[i]
            answer = answers.iloc[i]

            if context:
                context = '\n'.join([' '.join(z[1]) for z in context])
                prompt = f'Question: Given the information below, {question}\n{context}\nAnswer:'
            else:
                prompt = f'Question: {question}\nAnswer:'
            
            print(f'Prompt {i}: {prompt} {answer}')
            loss = llm(
                [prompt],
                [answer],
                None,
            )

            response = llm.inference(
                [prompt]
            )
            print(f'Answer {i}: {response}')

            print(f'loss {i}: {loss}')
            loss.backward()
            
            clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)

            if (i + 1) % grad_steps == 0:
                adjust_learning_rate(
                    optimizer.param_groups[0],
                    lr,
                    i / len(questions) + epoch,
                    epochs
                )

            optimizer.step()
            epoch_loss += float(loss)

            if (i + 1) % grad_steps == 0:
                lr = optimizer.param_groups[0]['lr']
        train_loss = epoch_loss / len(questions)
        print(f'Epoch {epoch}, Train Loss: {train_loss:4f}')

        print('Saving model...')
        fname = 'llm_direct_tuned.pt' if contexts is None else 'llm_tuned.pt'
        save_params_dict(llm, fname)
        print('Model saved successfully.')


def load_ner(name):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForTokenClassification.from_pretrained(name)
    return pipeline("ner", model=model, tokenizer=tokenizer, device=0, aggregation_strategy="max")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip_empty_vertices', required=False, default=True, action='store_true')
    parser.add_argument('--property_storage', type=str, required=False, default='managed')
    parser.add_argument('--fname', type=str, required=True)
    parser.add_argument('--truth_fname', type=str, required=True)
    parser.add_argument('--embeddings_dir', type=str, required=False)
    parser.add_argument('--embedding_type', type=str, required=False)
    parser.add_argument('--search_query', type=str, required=False)
    parser.add_argument('--w2v_path', required=False, type=str, default='./GoogleNews-vectors-negative300.bin.gz')
    parser.add_argument('--stage', required=False, type=str, default='test')
    parser.add_argument('--query_file', required=False, type=str)
    parser.add_argument('--query_search_file', required=False, type=str)
    parser.add_argument('--train_size', required=False, type=int)
    parser.add_argument('--seed', type=int, required=False, default=62)
    parser.add_argument('--ner_model', type=str, required=False, default="dslim/bert-large-NER")
    parser.add_argument('--llm_model', type=str, required=False, default='TinyLlama/TinyLlama-1.1B-Chat-v0.1')
    parser.add_argument('--llm_mparams', type=int, required=False, default=1)
    parser.add_argument('--gnn_params', type=str, required=False)
    parser.add_argument('--llm_params', type=str, required=False)
    parser.add_argument('--rag_type', type=str, required=False, default='direct')
    parser.add_argument('--num_epochs', type=int, required=False, default=1)
    parser.add_argument('--gnn_hidden_channels', type=int, required=False, default=1024)
    parser.add_argument('--gnn_num_layers', type=int, required=False, default=4)
    parser.add_argument('--lr', type=float, required=False, default=1e-5)
    parser.add_argument('--mlp_out_channels', type=int, required=False, default=2048) # 3200 for open_llama_3b_v2

    args = parser.parse_args()

    if args.stage in ['finetune_query', 'train', 'test', 'visualize']:
        if not args.embeddings_dir or not args.embedding_type:
            raise ValueError("Embeddings directory and embedding type must be provided for 'finetune_query', 'train', and 'test' stages.")

        eix, titles, sentences = read_wiki_data(
            args.fname,
            args.skip_empty_vertices
        )
        print('read wiki data')

        graph = BitGraph(
            'int64',
            'int64',
            'DEVICE',
            args.property_storage.upper(),
            'DEVICE',
        )

        graph.add_vertices(eix.max() + 1)
        graph.add_edges(eix[0], eix[1], 'link')

        start_time_emb = perf_counter()
        read_embeddings(
            graph,
            args.embeddings_dir,
            td=300 if args.embedding_type == 'w2v' else 1024,
            map_location='cpu' if args.property_storage.upper() in ['PINNED', 'HOST'] else 'cuda'
        )    
        end_time_emb = perf_counter()
        print(f'read embeddings into graph and built index, took {end_time_emb - start_time_emb} seconds')
        
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

    if args.search_query:
        qe = getem(args.search_query)
        vids = g.V().like('emb', [qe], 4).toArray()

        f = vids < len(titles)
        article_ids = vids[f]
        sentence_ids = vids[~f] - len(titles)

        print('articles:', titles.iloc[article_ids])
        print('sentences:', sentences.sentences.iloc[sentence_ids])

        print('query processed; exiting.')
        exit()
    
    gen_cpu = torch.Generator()
    gen_cpu.manual_seed(args.seed)

    truth_df = pandas.read_json(args.truth_fname)
    if args.train_size:
        perm = torch.randperm(len(truth_df), generator=gen_cpu)
        if args.stage == 'test':
            truth_df = truth_df.iloc[
                perm[args.train_size:2*args.train_size]
            ]
        else:
            truth_df = truth_df.iloc[
                perm[:args.train_size]
            ]

    supporting_facts = truth_df.supporting_facts
    contexts = truth_df.context
    questions = truth_df.question
    answers = truth_df.answer
    del truth_df

    if args.stage == 'finetune_llm':
        print(f'Fine tuning model "{args.llm_model}"')
        llm = LLM(model_name=args.llm_model, num_params=args.llm_mparams)
        if args.llm_params:
            load_params_dict(llm, args.llm_params)
        tune_llm(
            llm,
            questions,
            contexts if args.rag_type == 'direct' else None,
            answers,
            args.num_epochs,
            args.lr
        )
    elif args.stage == 'finetune_query':
        # Perform hyperparameter search
        # Write args to query file
        if not args.query_file or not args.query_search_file:
            raise ValueError("Must provide query file and query search file to perform query hyperparam tuning.")
        
        with open(args.query_search_file, 'r') as qf:
            query_grid = json.load(qf)
            query_grid.update({'random_state':args.seed})
        
        with torch.no_grad():
            tune_query(
                g,
                getem,
                load_ner(args.ner_model),
                questions,
                supporting_facts, 
                query_grid,
                args.query_file
            )
        exit()
    elif args.stage in ['train', 'test']:
        if not args.query_file:
            raise ValueError("Must provide query file if training with combined RAG")

        no_graph = (args.rag_type in ['direct', 'none'])
        
        llm = LLM(model_name=args.llm_model, num_params=args.llm_mparams)
        if args.llm_params:
            load_params_dict(llm, args.llm_params)
        else:
            warnings.warn("Fine tuning the LLM is recommended.")

        if no_graph:
            model = llm
        else:
            gnn = GAT(
                in_channels=1024,
                hidden_channels=args.gnn_hidden_channels,
                out_channels=1024,
                num_layers=args.gnn_num_layers,
                heads=4,
            )
            if args.gnn_params:
                load_params_dict(gnn, args.gnn_params)
            model = GRetriever(llm=llm, gnn=gnn, mlp_out_channels=args.mlp_out_channels)
        
        queries = pandas.read_json(args.query_file, lines=True)
        qp = queries.sort_values('avg_error').params.iloc[0]

        fn = train if args.stage == 'train' else test
        fn(
            model,
            load_ner(args.ner_model),
            g,
            getem,
            qp,
            titles,
            sentences,
            questions,
            answers,
            args.num_epochs,
            args.lr,
            args.rag_type
        )
    elif args.stage == 'visualize':
        import networkx as nx
        import matplotlib.pyplot as plt

        os.makedirs('./graphs', exist_ok=True)

        def decode(vids):
            names = []
            types = []
            for v in np.asarray(vids.cpu()).tolist():
                if v < len(titles):
                    names.append(str(titles.iloc[v]))
                    types.append('article')
                else:
                    names.append(str(sentences.sentences.iloc[v - len(titles)]))
                    types.append('sentence')

            return names, types
        
        ner = load_ner(args.ner_model)

        queries = pandas.read_json(args.query_file, lines=True)
        qp = queries.sort_values('avg_error').params.iloc[0]
        
        for i in range(len(questions)):            
            question = questions.iloc[i]
            answer = answers.iloc[i]

            prompt, data = get_prompt(ner, g, getem, qp, titles, sentences, question, rag_type='combined')
            eix = np.asarray(data.edge_index.cpu()).T

            names, types = decode(data.n_id)
            print(names)
            print(types)
            print(data.n_id)
            print(np.asarray(data.n_id.cpu())[eix])

            G = nx.DiGraph()
            G.add_nodes_from(np.arange(len(names)))
            for j in range(len(names)):
                G.nodes[j]['name'] = names[j]
                G.nodes[j]['ntype'] = types[j]

            G.add_edges_from(eix)

            nx.write_graphml(G, f'./graphs/question{i}.graphml')