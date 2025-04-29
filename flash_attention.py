import torch
import triton 
import triton.language as tl


@triton.jit
def _attn_fwd_inner(
    O_block,
    l_i,
    m_i,
    Q_block,
    K_block_ptr,
    V_block_ptr,
    block_index_q,
    softmax_scale,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
    offs_q: tl.constexpr,
    offs_kv: tl.constexpr,
    SEQ_LEN: tl.constexpr,
):
    # range of values handled by this stage
    if STAGE == 1:
        # left diagonal
        lo, hi = 0, block_index_q*BLOCK_SIZE_Q
        
    elif STAGE == 2:
        # used only for the block in which there is transition between non-masked and masked keys
        lo, hi = block_index_q* BLOCK_SIZE_Q, (block_index_q + 1)* BLOCK_SIZE_Q
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
    else:
        # only used for non-causal attention
        lo, hi = 0, SEQ_LEN
        
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    
    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)
        
        K_block = tl.load(K_block_ptr)
        QK_block = tl.dot(Q_block, K_block)
        
        if STAGE == 2:
            mask = offs_q[:, None] >= (start_kv + offs_kv[None, :])
            QK_block = QK_block * softmax_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
            QK_block = m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1) * softmax_scale)
            QK_block = QK_block*softmax_scale - m_ij[:, None]
            
        P_block = tl.math.exp(QK_block)
        l_ij = tl.sum(P_block, 1)
        alpha = tl.math.exp(m_i - m_ij)  # correction factor for previous l_i
        l_i = l_i* alpha + l_ij          # apply correction factor
        
        v_block = tl.load(V_block_ptr)
        P_block = P_block.to(tl.float16)
        
        O_block = O_block*alpha[:, None]
        O_block = tl.dot(P_block, v_block, O_block)  # O_block += P_block @ V_block
        
        m_i = m_ij
        
        # move to next block
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0)) # change in the seq len
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV)) # because of transpose deq_len comes as second
        
    return O_block, l_i, m_i 

@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages = num_stages,
            num_warps = num_warps,
        )
        for BLOCK_SIZE_Q in [64, 128]
        for BLOCK_SIZE_KV in [32, 64]
        for num_stages in ([3,4,7])
        for num_warps in [2,4]
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)


@triton.jit # THIS CHANGES THE USUAL PYTHON FUNCTION INTO TRITON KERNEL
def _attn_fwd(
    Q,
    K,
    V,
    softmax_scale,
    M,
    O,
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_O_batch,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)
    
    block_index_q = tl.program_id(0)
    index_batch_head = tl.program_id(1)
    index_batch = index_batch_head// NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    
    # Conditional logging for specific block
    if block_index_q == 0 and index_batch_head == 0:
        tl.device_print("Inside _attn_fwd: batch 0, head 0, Q block 0")
        start_token = block_index_q * BLOCK_SIZE_Q
        end_token = (block_index_q + 1) * BLOCK_SIZE_Q - 1
        tl.device_print("Processing Q tokens from: ", start_token)
        tl.device_print("Processing Q tokens to: ", end_token)
    
    qvk_offset = (
        index_batch.to(tl.int64)* stride_Q_batch
        + index_head.to(tl.int64) * stride_Q_head
    )
    
    Q_block_ptr = tl.make_block_ptr(    # Q[index_batch, index_head, block_index_q * BLOCK_SIZE_Q:, :]
        base= Q + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_Q_seq, stride_Q_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )
    
    V_block_ptr = tl.make_block_ptr(    # V[index_batch, index_head, :, :]
        base= V + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_V_seq, stride_V_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0),
    )
    
    K_block_ptr = tl.make_block_ptr(    # V[index_batch, index_head, :, :]
        base= K + qvk_offset,
        shape=(HEAD_DIM, SEQ_LEN),      # transposed
        strides=(stride_K_dim, stride_K_seq), # by inverting the strides the matrix can be transposed
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
        order=(0, 1),
    )
    
    O_block_ptr = tl.make_block_ptr(    # Q[index_batch, index_head, block_index_q * BLOCK_SIZE_Q:, :]
        base= O + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_O_seq, stride_O_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )
    
    # The offsets for hte tokens in the Q to process
    offs_q = block_index_q*BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    
    # The offset for the tokens in the K and V sequence to process
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)
    
    # The running maximum. One for eqch query
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")
    
    # The running sum. One per each query
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0
    
    # Tha accumulator for the output, which is a group of rows of the O matrix
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype= tl.float32)
    
    # load the blocks of Q to SRAM
    Q_block = tl.load(Q_block_ptr)
    
    
    if STAGE == 1 or STAGE == 3:
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            4 - STAGE,
            offs_q,
            offs_kv,
            SEQ_LEN,
        )
    
    if STAGE == 3:
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            2,
            offs_q,
            offs_kv,
            SEQ_LEN,
        )
    
    m_i += tl.math.log(l_i)
    O_block = O_block / l_i[:, None]
    m_ptrs = M + index_batch_head* SEQ_LEN + offs_q
    tl.store(m_ptrs, m_i)
    
@triton.jit
def _attn_bwd_preprocess(
    O, dO, D, SEQ_LEN, BLOCK_SIZE_Q: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    block_index_q = tl.program_id(0)
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    index_batch_head = tl.program_id(1)
    offs_dim = tl.arange(0, HEAD_DIM)
    
    if block_index_q == 0 and index_batch_head == 0:
        tl.device_print("Inside _attn_bwd_preprocess: batch 0, head 0, Q block 0")
    
    O_block = tl.load(
        O
        + index_batch_head * HEAD_DIM * SEQ_LEN
        + offs_q[:, None] * HEAD_DIM
        + offs_dim[None, :]
    ) 
    
    dO_block = tl.load(
        dO
        + index_batch_head * HEAD_DIM * SEQ_LEN
        + offs_q[:, None] * HEAD_DIM
        + offs_dim[None, :]
    ).to(tl.float32)
    
    D_block = tl.sum(dO_block*O_block, axis=1)
    
    #store
    D_block_ptrs = D + index_batch_head * SEQ_LEN + offs_q
    tl.store(D_block_ptrs, D_block)
    
@triton.jit
def _attn_bwd_dk_dv(
    Q,
    K,
    V,
    softmax_scale,
    dO,
    dQ,
    dK,
    dV,
    M,
    D,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
):
    index_batch_head = tl.program_id(2)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    offset_batch_head = (stride_batch*index_batch + stride_head*index_head).to(
        tl.int64
    )
    
    offset_batch_head_seq = (index_batch_head*SEQ_LEN).to(tl.int64)
    
    Q += offset_batch_head
    K += offset_batch_head
    V += offset_batch_head
    dO += offset_batch_head
    dQ += offset_batch_head
    dK += offset_batch_head
    dV += offset_batch_head
    
    M += offset_batch_head_seq
    D += offset_batch_head_seq
    
    offs_dim = tl.arange(0, HEAD_DIM)
    index_block_kv = tl.program_id(0)
    start_kv = index_block_kv * BLOCK_KV
    
    if index_block_kv == 0 and index_batch_head == 0:
        tl.device_print("Inside _attn_bwd_dk_dv: batch 0, head 0, KV block 0")
        start_kv_val = start_kv
        end_kv_val = start_kv + BLOCK_KV - 1
        tl.device_print("Processing KV tokens from: ", start_kv_val)
        tl.device_print("Processing KV tokens to: ", end_kv_val)
    
    # [0,0, start_kv: BLOCK_KV, HEAD_DIM ]
    offs_kv = start_kv + tl.arange(0, BLOCK_KV)
    
    dV_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)
    dK_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)
    
    K_block = tl.load(
        K + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    )
    
    # shape : [BLOCK_KV, HEAD_DIM]
    
    V_block = tl.load(
        V + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    )
    
    offs_q = tl.arange(0, BLOCK_Q)
    
    qT_ptrs = Q + offs_q[None, :] * stride_seq + offs_dim[:, None] * stride_dim
    dO_ptrs = dO + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    
    curr_q = 0
    num_steps = SEQ_LEN // BLOCK_Q
    for blk_idx in range(num_steps):
        qT_block = tl.load(qT_ptrs)
        offs_q = curr_q + tl.arange(0, BLOCK_Q)
        m = tl.load(M + offs_q)
        
        # (QK^T)^T = (K^T)^T(Q^T) = K(Q^T) = S^T
        QK_T_block = softmax_scale * tl.dot(K_block, qT_block)
        
        # Apply softmax using logsumexp methode
        P_T_block = tl.math.exp(QK_T_block - m[None, :])
        
        if STAGE == 3:
            mask_block = (offs_q[None, :] >= offs_kv[:, None])
            P_T_block = tl.where(mask_block, P_T_block, 0.0)
           
        dO_block = tl.load(dO_ptrs)
        dV_block += tl.dot(P_T_block.to(tl.float16), dO_block)
        
        Di = tl.load(D + offs_q)
        
        # dP = dO x V^T -> dP^T = V x dO^T
        dpT_block = tl.dot(V_block, tl.trans(dO_block)).to(tl.float32)
        
        dS_T_block = P_T_block * (dpT_block - Di[None, :])
        dS_T_block = dS_T_block.to(tl.float16)
        
        dK_block += softmax_scale * tl.dot(dS_T_block, tl.trans(qT_block))
        
        curr_q += BLOCK_Q
        qT_ptrs += BLOCK_Q * stride_seq
        dO_ptrs += BLOCK_Q * stride_seq
         
    dV_block_ptrs = dV + offs_kv[:, None] * stride_seq + offs_dim[None, :]* stride_dim
    tl.store(dV_block_ptrs, dV_block)
    
    dK_block_ptrs = dK + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dK_block_ptrs, dK_block)

@triton.jit
def _attn_bwd_dq(
    Q,
    K,
    V,
    softmax_scale,
    dO,
    dQ,
    dK,
    dV,
    M,
    D,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
):
    index_batch_head = tl.program_id(2)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    offset_batch_head = (stride_batch*index_batch + stride_head*index_head).to(
        tl.int64
    )
    
    offset_batch_head_seq = (index_batch_head*SEQ_LEN).to(tl.int64)
    
    Q += offset_batch_head
    K += offset_batch_head
    V += offset_batch_head
    dO += offset_batch_head
    dQ += offset_batch_head
    dK += offset_batch_head
    dV += offset_batch_head
    
    M += offset_batch_head
    D += offset_batch_head
    
    offs_dim = tl.arange(0, HEAD_DIM)
    index_block_kv = tl.program_id(0)
    start_q = index_block_kv * BLOCK_Q
    
    if index_block_kv == 0 and index_batch_head == 0:
        start_q_val = start_q
        end_q_val = start_q + BLOCK_KV - 1
        tl.device_print("Inside _attn_bwd_dq: batch 0, head 0, Q block 0")
        tl.device_print("Processing Q tokens from: ", start_q_val)
        tl.device_print("Processing Q tokens to: ", end_q_val)
    
    offs_q = start_q + tl.arange(0, BLOCK_Q)
    
    Q_block = tl.load(Q + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim)
    dQ_block = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)
    dO_block = tl.load(dO + offs_q[:, None]* stride_seq + offs_dim[None, :]* stride_dim)
    
    M_block = tl.load(M + offs_q)
    M_block = M_block[:, None]
    
    offs_kv = tl.arange(0, BLOCK_KV)
    
    # access the K, V as transposed blocks
    kT_ptrs = K + offs_kv[None, :] * stride_seq + offs_dim[:, None] * stride_dim
    vT_ptrs = V + offs_kv[None, :] * stride_seq + offs_dim[:, None] * stride_dim
    
    Di = tl.load(D + offs_q)
    
    curr_kv = 0
    num_steps = SEQ_LEN // BLOCK_KV
    
    for blk_idx in range(num_steps):
        K_T_block = tl.load(kT_ptrs)
        V_T_block = tl.load(vT_ptrs)
        
    
        QK_block = softmax_scale * tl.dot(Q_block, K_T_block)
        P_block = tl.math.exp(QK_block - M_block)
        
        if STAGE == 3:
            offs_kv = curr_kv + tl.arange(0, BLOCK_KV)
            mask_block = offs_q[:, None] >= offs_kv[None, :]
            P_block = tl.where(mask_block, P_block, 0.0)

        dP_block = tl.dot(dO_block, V_T_block).to(tl.float32)
        dS_block = P_block * (dP_block - Di[:, None])
        dS_block = dS_block.to(tl.float16)
        
        dQ_block += softmax_scale * tl.dot(dS_block, tl.trans(K_T_block))
        
        curr_kv += BLOCK_KV
        kT_ptrs += BLOCK_KV * stride_seq
        vT_ptrs += BLOCK_KV * stride_seq
        
    
        

class TritonAttention(torch.autograd. Function):
    
    @staticmethod
    def forward(ctx, Q, K, V, causal, softmax_scale):
        # ctx : contains necessary activation for backward pass
        
        HEAD_DIM_Q, HEAD_DIM_K  = Q.shape[-1], K.shape[-1]
        HEAD_DIM_V = V.shape[-1]
        
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape
        
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        
        O = torch.empty_like(Q)
        stage = 3 if causal else 1
        
        grid = lambda args: (
            # ceil(SEQ_LEN / BLOCK_SIZE_Q) = num of Q blocks
            triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]),  # which block of query is going to work
            BATCH_SIZE * NUM_HEADS,      # which HEAD and which batch is going to work
            1,   # z axis in the CUDA launch grid
        )
        
        # logsumexp for the backwardpass per query
        M = torch.empty(
            (BATCH_SIZE, HEAD_DIM, SEQ_LEN), device=Q.device, dtype=torch.float32
        )
        
        print(f"Forward pass start: Q shape {Q.shape}, K shape {K.shape}, V shape {V.shape}")

        # Timing the kernel launch
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        
        _attn_fwd[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=softmax_scale,
            M=M,
            O=O,
            stride_Q_batch=Q.stride(0),
            stride_Q_head=Q.stride(1),
            stride_Q_seq=Q.stride(2),
            stride_Q_dim=Q.stride(3),
            stride_K_batch=K.stride(0),
            stride_K_head=K.stride(1),
            stride_K_seq=K.stride(2),
            stride_K_dim=K.stride(3),
            stride_V_batch=V.stride(0),
            stride_V_head=V.stride(1),
            stride_V_seq=V.stride(2),
            stride_V_dim=V.stride(3),
            stride_O_batch=O.stride(0),
            stride_O_head=O.stride(1),
            stride_O_seq=O.stride(2),
            stride_O_dim=O.stride(3),
            BATCH_SIZE=Q.shape[0],
            NUM_HEADS=Q.shape[1],
            SEQ_LEN=Q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            STAGE=stage,
        )
        
        end_event.record()
        torch.cuda.synchronize()
        print(f"_attn_fwd kernel execution time: {start_event.elapsed_time(end_event)} ms")
        
        # Logging grid and chosen config
        chosen_config = _attn_fwd.best_config
        grid_dims = grid({'BLOCK_SIZE_Q': chosen_config.kwargs['BLOCK_SIZE_Q']})
        print(f"_attn_fwd grid dimensions: {grid_dims}")
        print(f"_attn_fwd chosen config: BLOCK_SIZE_Q={chosen_config.kwargs['BLOCK_SIZE_Q']}, BLOCK_SIZE_KV={chosen_config.kwargs['BLOCK_SIZE_KV']}, num_warps={chosen_config.num_warps}, num_stages={chosen_config.num_stages}")
        
        # Logging output shape
        print(f"Forward pass end: O shape {O.shape}")
        
        ctx.save_for_backward(Q, K, V, O, M)
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return O
    
    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, M = ctx.saved_tensors
        
        assert dO.is_continguous()
        assert Q.stride() == K.stride() == V.stride() == O.stride() == dO.stride()
        
        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)
        
        BATCH_SIZE, NUM_HEADS, SEQ_LEN = Q.shape[:3]
        NUM_WARPS, NUM_STAGES = 4, 3
        BLOCK_SIZE_MICRO, BLOCK_SIZE_MACRO = 32, 128
        
        preprocess_grid = (SEQ_LEN // BLOCK_SIZE_MACRO, BATCH_SIZE * NUM_HEADS)
        D = torch.empty_like(M)    # shape : [BATCH_SIZE, NUM_HEADS, SEQ_LEN]
        
        # Logging backward pass start
        print(f"Backward pass start: dO shape {dO.shape}")
        
        # Preprocess kernel
        print(f"Launching _attn_bwd_preprocess with grid {preprocess_grid}")
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        
        _attn_bwd_preprocess[preprocess_grid](
            O=O,
            dO=dO,
            D=D,
            SEQ_LEN=SEQ_LEN,
            BLOCK_SIZE_Q=BLOCK_SIZE_MACRO,
            HEAD_DIM=ctx.HEAD_DIM
        )
        
        end_event.record()
        torch.cuda.synchronize()
        print(f"_attn_bwd_preprocess kernel execution time: {start_event.elapsed_time(end_event)} ms")
        print(f"Preprocess output: D shape {D.shape}")
        
        grid = (SEQ_LEN // BLOCK_SIZE_MACRO, 1, BATCH_SIZE * NUM_HEADS)
        
        stage = 3 if ctx.causal else 1
        
        # dK, dV kernel
        print(f"Launching _attn_bwd_dk_dv with grid {grid}")
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        
        # fix KV and iterate through all the Q blocks
        _attn_bwd_dk_dv[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=ctx.softmax_scale,
            dO=dO,
            dQ=dQ,
            dK=dK,
            dV=dV,
            M=M,
            D=D,
            stride_batch = Q.stride(0),
            stride_head = Q.stride(1),
            stride_seq = Q.stride(2),
            stride_dim = Q.stride(3),
            NUM_HEADS = NUM_HEADS,
            SEQ_LEN = SEQ_LEN,
            BLOCK_Q = BLOCK_SIZE_MICRO,
            BLOCK_KV = BLOCK_SIZE_MACRO,
            HEAD_DIM = ctx.HEAD_DIM,
            STAGE = stage,
            num_warps = NUM_WARPS,
            num_stages = NUM_STAGES,         
        )
        
        end_event.record()
        torch.cuda.synchronize()
        print(f"_attn_bwd_dk_dv kernel execution time: {start_event.elapsed_time(end_event)} ms")
        
        
        # dQ kernel
        print(f"Launching _attn_bwd_dq with grid {grid}")
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        
        # fix Q and iterate through all the KV block
        _attn_bwd_dq[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=ctx.softmax_scale,
            dO=dO,
            dQ=dQ,
            dK=dK,
            dV=dV,
            M=M,
            D=D,
            stride_batch = Q.stride(0),
            stride_head = Q.stride(1),
            stride_seq = Q.stride(2),
            stride_dim = Q.stride(3),
            NUM_HEADS = NUM_HEADS,
            SEQ_LEN = SEQ_LEN,
            BLOCK_Q = BLOCK_SIZE_MICRO,
            BLOCK_KV = BLOCK_SIZE_MACRO,
            HEAD_DIM = ctx.HEAD_DIM,
            STAGE = stage,
            num_warps = NUM_WARPS,
            num_stages = NUM_STAGES,         
        )
        
        end_event.record()
        torch.cuda.synchronize()
        print(f"_attn_bwd_dq kernel execution time: {start_event.elapsed_time(end_event)} ms")
        
        # Logging output shapes
        print(f"Backward pass end: dQ shape {dQ.shape}, dK shape {dK.shape}, dV shape {dV.shape}")
        
        return dQ, dK, dV, None, None
        

# forward pass 
def test_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16):
    Q = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    K = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    
    V = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    
    softmax_scale = 1 / (HEAD_DIM)**0.5
    
    do = torch.rand_like(Q)  # used for backward pass
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    
    MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device="cuda"))
    P = torch.matmul(Q, K.transpose(2,3)) * softmax_scale
    if causal:
        P[:, :, MASK == 0] = float("-inf")
        
    P = torch.softmax(P.float(), dim=1).half()
    ref_o = torch.matmul(P, V)
    
    end_event.record()
    torch.cuda.synchronize()
    pytorch_forward_time = start_event.elapsed_time(end_event)
    print(f"PyTorch Forward Time: {pytorch_forward_time} ms")
    
    start_event.record()
    ref_o.backward(do)
    end_event.record()
    torch.cuda.synchronize()
    pytorch_backward_time = start_event.elapsed_time(end_event)
    print(f"PyTorch Backward Time: {pytorch_backward_time} ms")
    
    ref_dv, V.grad = V.grad.clone(), None
    ref_dk, K.grad = K.grad.clone(), None
    ref_dq, Q.grad = Q.grad.clone(), None
    
    
    # Triton Impplementation
    tri_out = TritonAttention.apply(Q, K,V, causal, softmax_scale).half()
    tri_out.backward(do)
    tri_dv, V.grad = V.grad.clone(), None
    tri_dk, K.grad = K.grad.clone(), None
    tri_dq, Q.grad = Q.grad.clone(), None   
 
    rtol = 0.0       #  relative tolerance
    atol = 1e-2      #  absolute tolerance
    assert torch.allclose(ref_o, tri_out, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dv, tri_dv, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dk, tri_dk, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dq, tri_dq, atol=atol, rtol=rtol)

    
if __name__ == "__main__":
    test_op(BATCH_SIZE=8, NUM_HEADS=16, SEQ_LEN=4096, HEAD_DIM=64, causal=True)
    test_op(BATCH_SIZE=8, NUM_HEADS=16, SEQ_LEN=4096, HEAD_DIM=64, causal=False)
    print("success")