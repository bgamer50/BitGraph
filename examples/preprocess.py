import os
import argparse
import warnings
from time import perf_counter

import torch
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer

torch.set_float32_matmul_precision('high')

class Sentence_Transformer(torch.nn.Module):
    """
    Adapted from the version in G-Retriever (https://github.com/XiaoxinHe/G-Retriever)
    """

    def __init__(self, bert_model):
        super(Sentence_Transformer, self).__init__()
        self.bert_model = bert_model

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        data_type = token_embeddings.dtype
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(data_type)
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids, att_mask):
        bert_out = self.bert_model(input_ids=input_ids, attention_mask=att_mask)
        sentence_embeddings = self.mean_pooling(bert_out, att_mask)

        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

class Word2Vec_Transformer(torch.nn.Module):
    """
    Adapted from the version in G-Retriever (https://github.com/XiaoxinHe/G-Retriever)
    """

    def __init__(self, w2v_model, dim):
        super(Word2Vec_Transformer, self).__init__()
        self.w2v_model = w2v_model
        self.dim = dim
    
    def forward(self, text):
        vecs = []
        for word in text.split():
            try:
                vecs.append(self.w2v_model[word])
            except KeyError:
                pass
        
        if len(vecs) > 0:
            emb = torch.tensor(sum(vecs) / len(vecs))
        else:
            emb = torch.zeros(self.dim, dtype=torch.float32)
        
        return emb

def worker_init(rank, world_size):
    import rmm
    rmm.reinitialize(devices=rank)

    torch.cuda.set_device(rank)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12354'
    torch.distributed.init_process_group('nccl', rank=rank, world_size=world_size,)


def write_emb(dir_out, rank, index, emb):
    out_fname = os.path.join(
        dir_out,
        f'part_{rank}_{index}.pt',
    )

    e = torch.concat(emb)
    torch.save(e.detach().contiguous(), out_fname)
    return index + 1


def batch_fwd(module, mask, input_id, shape):
    mask = torch.concat(mask).cuda()
    input_id = torch.concat(input_id).cuda()
    shape = torch.tensor(shape, dtype=torch.int32, device='cuda').cumsum(0)

    emb = []
    for s in range(shape.shape[0]):
        ix = input_id[:shape[0]] if s == 0 else input_id[shape[s-1] : shape[s]]
        mx = mask[:shape[0]] if s == 0 else mask[shape[s-1] : shape[s]]

        ix = ix.reshape((1,ix.shape[0]))
        mx = mx.reshape((1,mx.shape[0]))

        #print(ix, mx)

        a = module(ix, mx)
        emb.append(a)
    
    return emb


def encode_n2v(module, tokenizer, strings, batch_size, out_path, start_part):
    assert tokenizer is None

    rank = torch.distributed.get_rank()

    start_time = perf_counter()
    with torch.no_grad():
        emb = []
        index = start_part
        for i, a in enumerate(strings):
            if a is None:
                a = "None"
                
            emb.append(module(a))

            if i % batch_size == 0:
                ttm = perf_counter() - start_time

                elt = perf_counter() - start_time
                start_time = perf_counter()
                print(f'rank: {rank}, title index: {i}, tok time: {ttm}s total time: {elt}s')

            if i % 1_000_000 == 0:
                index = write_emb(out_path, rank, index, emb)
                emb = []
        
        write_emb(out_path, rank, index, emb)
        del emb


def encode(module, tokenizer, strings, batch_size, out_path, start_part):
    if tokenizer is None:
        encode_n2v(module, tokenizer, strings, batch_size, out_path, start_part)
        return

    rank = torch.distributed.get_rank()

    start_time = perf_counter()
    with torch.no_grad():
        mask = []
        input_id = []
        shape = []
        emb = []
        index = start_part
        for i, a in enumerate(strings):
            if a is None:
                a = "None"
                
            t = tokenizer(a, return_tensors='pt')
            while t.input_ids.shape[1] > 512:
                a = a[:-10]
                t = tokenizer(a, return_tensors='pt')

            input_id.append(t.input_ids.reshape((t.input_ids.shape[1],)))
            mask.append(t.attention_mask.reshape((t.attention_mask.shape[1],)))
            shape.append(t.input_ids.shape[1])

            if i % batch_size == 0:
                ttm = perf_counter() - start_time
                emb += batch_fwd(module, mask, input_id, shape)

                elt = perf_counter() - start_time
                start_time = perf_counter()
                print(f'rank: {rank}, title index: {i}, tok time: {ttm}s total time: {elt}s')

                mask = []
                input_id = []
                shape = []

            if i % 1_000_000 == 0:
                index = write_emb(out_path, rank, index, emb)
                emb = []
        

        emb += batch_fwd(module, mask, input_id, shape)
        write_emb(out_path, rank, index, emb)
        del emb


def run_preprocess(rank, world_size, fname_in, dir_out, model, tokenizer, start_ix, start_part, node_type, batch_size=1024):
    worker_init(rank, world_size)
    if tokenizer is None:
        module = model
    else:
        model = model.cuda()
        mod = Sentence_Transformer(model).cuda()

        import torch._dynamo
        torch._dynamo.reset()

        module = torch.compile(mod, fullgraph=True)

    title_path = os.path.join(dir_out, 'titles')
    os.makedirs(title_path, exist_ok=True)

    sentence_path = os.path.join(dir_out, 'sentences')
    os.makedirs(sentence_path, exist_ok=True)

    import torch, cudf
    df = cudf.read_json(fname_in, lines=True)
    ix = torch.tensor_split(torch.arange(len(df)), world_size)[rank]
    df = df.iloc[ix]

    if node_type == 'title':
        encode(module, tokenizer, df.title.to_pandas()[start_ix:], batch_size, title_path, start_part)
    elif node_type == 'sentence':
        sentences = df.sentences.explode().reset_index().rename({"index": 'article'},axis=1).sentences.to_pandas()[start_ix:]
        del df
        encode(module, tokenizer, sentences, batch_size, sentence_path, start_part)
    else:
        raise ValueError("Invalid node type.  Expected 'article' or 'sentence'.")
    
    print(f'rank {rank} completed!')
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname_in', required=True, type=str)
    parser.add_argument('--dir_out', required=True, type=str)
    parser.add_argument('--start_ix', required=False, type=int, default=0)
    parser.add_argument('--start_part', required=False, type=int, default=0)
    parser.add_argument('--node_type', required=True, type=str)
    parser.add_argument('--encoder', required=False, type=str, default='w2v')
    parser.add_argument('--w2v_path', required=False, type=str, default='./GoogleNews-vectors-negative300.bin.gz')
    args = parser.parse_args()

    if args.encoder == 'roberta':
        model = AutoModel.from_pretrained('sentence-transformers/all-roberta-large-v1')
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-roberta-large-v1')
    elif args.encoder == 'w2v':
        import gensim
        warnings.warn("Word2Vec encoder is for testing/debugging purposes only!")
        model = Word2Vec_Transformer(
            gensim.models.KeyedVectors.load_word2vec_format(args.w2v_path, binary=True),
            dim=300,
        )
        tokenizer = None
    else:
        raise ValueError("Invalid encoder.  Valid options are w2v and roberta.")

    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(
        run_preprocess,
        args = (
            world_size,
            args.fname_in,
            args.dir_out,
            model,
            tokenizer,
            args.start_ix,
            args.start_part,
            args.node_type,
        ),
        nprocs=world_size,
    )