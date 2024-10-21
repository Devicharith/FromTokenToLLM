from dataclasses import dataclass
import torch.nn as nn
from torch.nn import functional as F
import math
import tiktoken
import torch
import time
import inspect
# import code; code.interact(local = locals())


max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_iters = 25
lr_decay_iters = 85
max_iters = 100
def cosine_lr(it):
    if it<warmup_iters:
        lr = max_lr * ((it+1)/warmup_iters)
        return lr
    elif it>lr_decay_iters:
        return min_lr
    else:
        decay_rate = (it-warmup_iters)/(max_iters-warmup_iters) # Value between 0 to 1 and mesures how far is it from the warmpups
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_rate))
        return min_lr + coeff * (max_lr-min_lr)

class DataLoarderLite:
    def __init__(self, file, batches, tokens):
        with open(file) as op:
            data = op.read()
        
        enc = tiktoken.get_encoding('gpt2')
        self.tokens = torch.tensor(enc.encode(data))
            
        self.b = batches
        self.t = tokens
        self.batch_num = 0
    
    def next_batch(self):
        batch = self.tokens[self.batch_num: self.batch_num+(self.b*self.t)+1]
        inputs = batch[:-1].view(self.b, self.t)
        targets = batch[1:].view(self.b, self.t)

        self.batch_num += self.b*self.t
        if self.batch_num + (self.b*self.t+1) > len(self.tokens):
            self.batch_num = 0
        
        return inputs, targets


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        # output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        # regularization
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        
        self.c_proj.SCALE_INIT = 1

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    
        # manual implementation of attention
        # this materializes the large (T,T) matrix for all the queries and keys
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = F.scaled_dot_product_attention(q,k,v, is_causal = True)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4*config.n_embed)
        self.gelu = nn.GELU(approximate = "tanh")
        self.c_proj = nn.Linear(4*config.n_embed, config.n_embed)
        self.c_proj.SCALE_INIT = 1
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp =  MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024  # Size of the input sequence or num tokens for the model
    vocab_size: int = 50257  # Size of the vocabulary (number of unique tokens)
    n_layer: int = 12  # Number of transformer layers in the model
    n_head: int = 12  # Number of attention heads in each layer
    n_embed: int = 768  # Embedding dimension (hidden size) for the modelca
    batch_size: int = 4 # Batch_size for training
    mode: str = "training" # Training/eval modes

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embed),
                "wpe": nn.Embedding(config.block_size, config.n_embed),
                "h":   nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f":nn.LayerNorm(config.n_embed)
            }
        )

        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias = False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # initalizations
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "SCALE_INIT"):
                std *= (2*self.config.n_layer)**(-0.5)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets = None):
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        pos = torch.arange(0, t, device=idx.device)
        embds = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)

        x = embds + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x_norm = self.transformer.ln_f(x)

        #logits
        logits = self.lm_head(x_norm)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def configure_optimizer(self, lr, betas, weight_decay, device_name):
        # Pull out only required weights
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # Create optims group with 2D groups other else
        decay_params = [p for pm, p in param_dict.items() if p.dim()>=2]
        nondecay_params = [p for pm, p in param_dict.items() if p.dim()<=1]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nondecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nondecay_params)}, with {num_nodecay_params:,} parameters")

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nondecay_params, "weight_decay": 0.0}
        ]

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_name == 'cuda'
        print(f"using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas, fused=use_fused)
        return optimizer


#----------------------- Training and Validation --------------------------------

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available! 🚀")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Device count: {torch.cuda.device_count()}")
    device = "cuda"
else:
    print("CUDA is not available.")
    device = "cpu"

torch.manual_seed(37)
if torch.cuda.is_available():
    torch.cuda.manual_seed(37)

# Load the LLM configurations
config = GPTConfig(vocab_size = 50304)

# load the LLM model using the configurations
model = GPT(config)

# load the model on to the GPU/CPU
model.to(device)
# model = torch.compile(model, mode="reduce-overhead")

# Use low percision for matmul
torch.set_float32_matmul_precision(precision="high")

# Gradient Accumilation
tokens_per_batch = 16384 # 524288
B = config.batch_size
T = config.block_size
assert tokens_per_batch % (B*T) == 0
grads_accum_steps = tokens_per_batch//(B*T)

if config.mode=="training":
    # Load the training data
    data_loader = DataLoarderLite("shakespere.txt", config.batch_size, config.block_size)

    optimizer = model.configure_optimizer(lr=1e-3, betas = (0.9, 0.95), weight_decay=0.1, device_name=device)

    for step in range(max_iters):
        t0 = time.time()
        accum_grad = 0.0
        for grad_steps in range(grads_accum_steps):
            optimizer.zero_grad()
            inputs, targets = data_loader.next_batch()
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.autocast(device_type="cuda", dtype = torch.bfloat16):
                logits, loss = model(inputs, targets)

            # we have to scale the loss to account for gradient accumulation,
            # because the gradients just add on each successive backward().
            # addition of gradients corresponds to a SUM in the objective, but
            # instead of a SUM we want MEAN, so we scale the loss here
            loss = loss/grads_accum_steps
            accum_grad += loss.detach()
            loss.backward()

        # Grad clipping before update
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Cosine lr 
        lr = cosine_lr(step)
        for params in optimizer.param_groups:
            params["lr"] = lr

        # Grad Update
        optimizer.step()

        torch.cuda.synchronize() # Wait for the GPU to finish the tasks

        t1 = time.time()
        tps = (config.batch_size * config.block_size)/(t1-t0)
        print(f"{step}th iter, loss: {accum_grad.item()}, clipping: {norm:.4f}, lr: {lr:.4e}, time taken {t1-t0:.2f}s, tps: {tps:.4f}")

else:
    model.eval()
    while x.size(-1)<15:
        st = time.time()

        inputs, targets = data_loader.next_batch()
        logits, loss = model(inputs, targets)

        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim = -1)

        topK_probs, topK_indices = torch.topk(probs, 50, dim=-1)
        # print(topK_probs[:5], topK_indices[:5])

        ix = torch.multinomial(topK_probs, 1)
        # print(ix)

        xcol = torch.gather(topK_indices, -1, ix)
        # print(xcol)

        x = torch.cat((x, xcol), dim=-1)

    for i in x:
        tokens = i.tolist()
        decoded = tokenizer.decode(tokens)
        print(decoded)

    time_taken = time.time()-st
    print(f"Time taken to run the code: {time_taken:.6f} seconds")
