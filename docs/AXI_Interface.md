# AXI4-Full Interface Details
## DVCon India 2026 – Parallel BRAM-Based Affinity Scorer

---

## 1. Interface Overview

The accelerator exposes a single **AXI4-Full slave** port. The CPU (VEGA RISC-V / Zynq PS) acts as the AXI master.

| Property | Value |
|----------|-------|
| AXI Version | AXI4-Full (AMBA AXI4) |
| Data Width | 32-bit |
| Address Width | 32-bit |
| ID Width | 4-bit |
| Burst Types | INCR (type 01) |
| Max Burst Length | 16 beats (AWLEN/ARLEN up to 15) |
| Beat Size | 4 bytes (AWSIZE/ARSIZE = 3'b010) |
| Clock | Single clock domain (ACLK) |
| Reset | Active-LOW synchronous (ARESETn) |

---

## 2. Signal Table

### Global
| Signal | Dir | Width | Description |
|--------|-----|-------|-------------|
| ACLK | I | 1 | AXI clock |
| ARESETn | I | 1 | Active-low reset |

### Write Address Channel (AW)
| Signal | Dir | Width | Description |
|--------|-----|-------|-------------|
| AWID | I | 4 | Transaction ID |
| AWADDR | I | 32 | Start byte address |
| AWLEN | I | 8 | Burst length minus 1 |
| AWSIZE | I | 3 | Beat size (010=4B) |
| AWBURST | I | 2 | Burst type (01=INCR) |
| AWVALID | I | 1 | Address valid |
| AWREADY | O | 1 | Slave ready |

### Write Data Channel (W)
| Signal | Dir | Width | Description |
|--------|-----|-------|-------------|
| WDATA | I | 32 | Write data |
| WSTRB | I | 4 | Byte enable |
| WLAST | I | 1 | Last beat flag |
| WVALID | I | 1 | Data valid |
| WREADY | O | 1 | Slave ready |

### Write Response Channel (B)
| Signal | Dir | Width | Description |
|--------|-----|-------|-------------|
| BID | O | 4 | Matching transaction ID |
| BRESP | O | 2 | 00=OKAY |
| BVALID | O | 1 | Response valid |
| BREADY | I | 1 | Master ready |

### Read Address Channel (AR)
| Signal | Dir | Width | Description |
|--------|-----|-------|-------------|
| ARID | I | 4 | Transaction ID |
| ARADDR | I | 32 | Start byte address |
| ARLEN | I | 8 | Burst length minus 1 |
| ARSIZE | I | 3 | Beat size |
| ARBURST | I | 2 | Burst type |
| ARVALID | I | 1 | Address valid |
| ARREADY | O | 1 | Slave ready |

### Read Data Channel (R)
| Signal | Dir | Width | Description |
|--------|-----|-------|-------------|
| RID | O | 4 | Matching transaction ID |
| RDATA | O | 32 | Read data |
| RRESP | O | 2 | 00=OKAY |
| RLAST | O | 1 | Last beat flag |
| RVALID | O | 1 | Data valid |
| RREADY | I | 1 | Master ready |

---

## 3. Register Map

All addresses are byte addresses relative to the AXI slave base.

| Byte Offset | Register | Access | Width | Description |
|-------------|----------|--------|-------|-------------|
| 0x0000 | CTRL | W | 32 | `[0]`=START pulse; write 1 to begin |
| 0x0004 | STATUS | R | 32 | `[0]`=DONE, `[1]`=BUSY |
| 0x0008 | N_OBJECTS | RW | 32 | `[15:0]` number of objects (max MAX_OBJECTS) |
| 0x000C | BEST_OBJECT | R | 32 | `[15:0]` winning object index |
| 0x0010 | BEST_SCORE | R | 32 | `[31:0]` best final score (Q1.14 fixed-point) |
| 0x0100 + i×4 | TASK_EMB[i] | RW | 32 | `[7:0]` task embedding byte i (Q2.6 signed), i=0..EMB_DIM-1 |
| 0x0200 + o×4 | CONF[o] | RW | 32 | `[15:0]` detection confidence for object o (Q1.14 signed) |
| 0x0300 + o×4 | BOOST[o] | RW | 32 | `[15:0]` domain boost for object o (Q1.14 signed) |
| 0x0400 + o×4 | PENALTY[o] | RW | 32 | `[0]` penalty flag for object o |
| 0x1000 + (o×EMB_DIM+b)×4 | OBJ_EMB[o][b] | RW | 32 | `[7:0]` label embedding byte b for object o (Q2.6 signed) |

> **Note:** The address map is unchanged from the serial design. CPU software does not need modification. The backend storage is now BRAM-backed (replicated across N_LANES for parallel read access).

---

## 4. Storage Backend

| Data | Storage Type | Details |
|------|-------------|---------|
| Object embeddings | **Block RAM** (BRAM) | Dual-port, N_LANES replicas, byte-write-enable |
| Confidence/Boost/Penalty | Distributed RAM (LUT registers) | Inside parallel_scorer |
| Task embedding | Flip-flop registers | Shared across all lanes |
| CSRs | Flip-flop registers | In axi4_affinity_top |

### BRAM Replicas

Each of the N_LANES scoring lanes has its own BRAM copy of all object embeddings. This enables true parallel reads (one per lane) without port contention.

- **Write**: AXI writes are broadcast to all N_LANES BRAM replicas simultaneously
- **Read (compute)**: Each lane reads from its own BRAM Port B independently
- **Read (AXI readback)**: Replica 0 Port A provides read data for AXI read responses

### BRAM Sizing

| EMB_DIM | Width (bits) | Depth | Size per replica | N_LANES=4 total |
|---------|-------------|-------|-------------------|-----------------|
| 8 | 64 | 16 | 1,024 bits | 4,096 bits (< 1 BRAM36) |
| 8 | 64 | 80 | 5,120 bits | 20,480 bits (< 1 BRAM36) |
| 512 | 4,096 | 16 | 65,536 bits | 262,144 bits (~8 BRAM36) |
| 512 | 4,096 | 80 | 327,680 bits | 1,310,720 bits (~36 BRAM36) |

---

## 5. Fixed-Point Format

| Signal | Format | Range | Scale |
|--------|--------|-------|-------|
| Embedding bytes | Q2.6 (8-bit signed) | -2.0 to +1.984 | ÷64 |
| Confidence | Q1.14 (16-bit signed) | 0.0 to 1.0 | ÷16384 |
| Boost | Q1.14 (16-bit signed) | 0.0 to 0.5 | ÷16384 |
| BEST_SCORE | Q1.14 (32-bit) | 0.0 to 1.0 | ÷16384 |

---

## 6. Programming Sequence

```
1. Assert ARESETn = 0 for ≥8 clock cycles, then de-assert (=1)
2. Write N_OBJECTS to 0x0008
3. Burst-write TASK_EMB bytes to 0x0100 (AWLEN = EMB_DIM-1)
4. For each object o = 0..N-1:
   a. Write CONF[o]    to 0x0200 + o×4
   b. Write BOOST[o]   to 0x0300 + o×4
   c. Write PENALTY[o] to 0x0400 + o×4
   d. Burst-write OBJ_EMB[o] to 0x1000 + o×EMB_DIM×4 (AWLEN = EMB_DIM-1)
5. Write 0x1 to CTRL (0x0000) → triggers START
6. Poll STATUS (0x0004) until bit[0]=1 (DONE)
7. Read BEST_OBJECT from 0x000C
8. Read BEST_SCORE  from 0x0010
```

> This sequence is identical to the serial design. The BRAM backing is transparent to software.

---

## 7. Timing

| Operation | Latency |
|-----------|---------|
| AXI write (single) | 3–4 clock cycles |
| AXI read (single) | 3–5 clock cycles (1 extra for BRAM pipeline) |
| Per-batch compute (N_LANES objects) | 9 cycles |
| Per-object (amortised, N_LANES=4) | ~2.25 cycles |
| N=2 objects total | ~9 cycles |
| N=80 objects (COCO full, N_LANES=4) | ~180 cycles @ 100 MHz = 1.8 µs |

---

## 8. AXI4-Full vs AXI4-Lite Distinction

| Feature | AXI4-Lite | AXI4-Full (this design) |
|---------|-----------|-------------------------|
| Burst support | No (AWLEN=0 only) | Yes (AWLEN 0–15) |
| ID signals | No | Yes (4-bit AWID/ARID) |
| WLAST signal | Not required | Yes (required) |
| Transfer size (AWSIZE) | Fixed 32-bit | Signalled per transaction |

Burst writes are used to programme all EMB_DIM embedding bytes in a single AXI transaction.
