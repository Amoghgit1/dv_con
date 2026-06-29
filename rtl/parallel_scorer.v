`timescale 1ns / 1ps
// =============================================================
//  parallel_scorer.v – N-Lane Parallel Scoring Engine with BRAM
//  DVCon India 2026
//
//  Architecture:
//    • N_LANES replicated obj_emb_bram instances (broadcast writes)
//    • N_LANES scoring_lane instances (MAC + score pipeline each)
//    • Internal metadata register arrays (conf, boost, penalty)
//    • Dispatch FSM: loads all lanes in parallel from BRAM, fires
//      all lanes simultaneously, reduces to find the batch max,
//      and updates the running best across rounds.
//
//  Per-batch timing:
//    LOAD(1) + PIPE(1) + LATCH(1) + FIRE(1) + WAIT(4) + REDUCE(1)
//    = 9 cycles per batch of N_LANES objects.
//
//  Throughput:  9 / N_LANES  cycles per object
//              (2.25 cyc/obj for N_LANES=4  vs  8 cyc/obj serial)
// =============================================================
module parallel_scorer #(
    parameter EMB_DIM     = 8,
    parameter MAX_OBJECTS = 16,
    parameter N_LANES     = 4
)(
    input  wire        clk,
    input  wire        rst,             // active-high synchronous reset

    // ── Control ──────────────────────────────────────────────
    input  wire        start,           // 1-cycle pulse
    input  wire [15:0] n_objects,

    // ── Task embedding (shared, register-based) ──────────────
    input  wire signed [8*EMB_DIM-1:0] task_emb_flat,

    // ── Embedding BRAM write interface (broadcast to replicas)─
    input  wire [EMB_DIM-1:0]                emb_wr_we,
    input  wire [$clog2(MAX_OBJECTS)-1:0]    emb_wr_addr,
    input  wire [8*EMB_DIM-1:0]              emb_wr_din,

    // ── Embedding BRAM read interface (AXI readback, replica 0)
    input  wire [$clog2(MAX_OBJECTS)-1:0]    emb_axi_rd_addr,
    output wire [8*EMB_DIM-1:0]              emb_axi_rd_dout,

    // ── Metadata write interface ─────────────────────────────
    input  wire                              meta_conf_we,
    input  wire                              meta_boost_we,
    input  wire                              meta_penalty_we,
    input  wire [$clog2(MAX_OBJECTS)-1:0]    meta_wr_addr,
    input  wire signed [15:0]                meta_wr_conf,
    input  wire signed [15:0]                meta_wr_boost,
    input  wire                              meta_wr_penalty,

    // ── Metadata read interface (AXI readback) ───────────────
    input  wire [$clog2(MAX_OBJECTS)-1:0]    meta_rd_addr,
    output wire signed [15:0]                meta_rd_conf,
    output wire signed [15:0]                meta_rd_boost,
    output wire                              meta_rd_penalty,

    // ── Results ──────────────────────────────────────────────
    output reg         done,
    output reg  [15:0] best_object,
    output reg  signed [31:0] best_score
);

    // =========================================================
    //  Derived parameters
    // =========================================================
    localparam OBJ_IDX_W = $clog2(MAX_OBJECTS);
    // LANE_W must hold values 0 .. N_LANES (inclusive)
    localparam LANE_W    = $clog2(N_LANES + 1);

    // =========================================================
    //  FSM States
    // =========================================================
    localparam PS_IDLE  = 4'd0;
    localparam PS_SETUP = 4'd1;
    localparam PS_LOAD  = 4'd2;   // set BRAM Port-B addresses for all lanes
    localparam PS_PIPE  = 4'd3;   // wait 1 cycle for BRAM output to register
    localparam PS_LATCH = 4'd4;   // capture BRAM outputs + metadata into lanes
    localparam PS_FIRE  = 4'd5;   // pulse start on all lanes
    localparam PS_WAIT  = 4'd6;   // wait for all lane pipelines to complete
    localparam PS_REDUCE= 4'd7;   // compare batch max with running best
    localparam PS_DONE  = 4'd8;

    reg [3:0] ps_state;

    // =========================================================
    //  Control registers
    // =========================================================
    reg [15:0]        n_obj_latched;
    reg [15:0]        batch_base;       // first object ID of current batch
    reg [LANE_W-1:0]  n_active;         // lanes active this batch (≤ N_LANES)
    reg               lane_fire;        // start pulse broadcast to all lanes

    // =========================================================
    //  Per-lane BRAM read ports
    // =========================================================
    reg  [OBJ_IDX_W-1:0]    lane_bram_addr [0:N_LANES-1];
    wire [8*EMB_DIM-1:0]    lane_bram_dout [0:N_LANES-1];

    // =========================================================
    //  Per-lane input registers (latched from BRAM + metadata)
    // =========================================================
    reg signed [8*EMB_DIM-1:0] lane_emb     [0:N_LANES-1];
    reg signed [15:0]          lane_conf    [0:N_LANES-1];
    reg signed [15:0]          lane_boost   [0:N_LANES-1];
    reg                        lane_penalty [0:N_LANES-1];
    reg [15:0]                 lane_obj_id  [0:N_LANES-1];

    // =========================================================
    //  Lane outputs
    // =========================================================
    wire signed [31:0] lane_score [0:N_LANES-1];
    wire               lane_done  [0:N_LANES-1];

    // =========================================================
    //  Running best across all batches
    // =========================================================
    reg signed [31:0] run_best_score;
    reg [15:0]        run_best_obj;

    // =========================================================
    //  Metadata register arrays (internal storage)
    // =========================================================
    (* ram_style = "distributed" *)
    reg signed [15:0] conf_mem    [0:MAX_OBJECTS-1];
    (* ram_style = "distributed" *)
    reg signed [15:0] boost_mem   [0:MAX_OBJECTS-1];
    reg               penalty_mem [0:MAX_OBJECTS-1];

    // ── Metadata writes (from AXI) ───────────────────────────
    integer mw;
    always @(posedge clk) begin
        if (rst) begin
            for (mw = 0; mw < MAX_OBJECTS; mw = mw + 1) begin
                conf_mem[mw]    <= 16'sd0;
                boost_mem[mw]   <= 16'sd0;
                penalty_mem[mw] <= 1'b0;
            end
        end else begin
            if (meta_conf_we)    conf_mem   [meta_wr_addr] <= meta_wr_conf;
            if (meta_boost_we)   boost_mem  [meta_wr_addr] <= meta_wr_boost;
            if (meta_penalty_we) penalty_mem[meta_wr_addr] <= meta_wr_penalty;
        end
    end

    // ── Metadata reads (AXI readback) ────────────────────────
    assign meta_rd_conf    = conf_mem   [meta_rd_addr];
    assign meta_rd_boost   = boost_mem  [meta_rd_addr];
    assign meta_rd_penalty = penalty_mem[meta_rd_addr];

    // =========================================================
    //  BRAM instances – N_LANES replicas (broadcast writes)
    // =========================================================
    //  Port A address mux: during writes, use emb_wr_addr;
    //  otherwise, use emb_axi_rd_addr (for AXI readback).
    wire               bram_a_we_any = |emb_wr_we;
    wire [OBJ_IDX_W-1:0] bram_a_addr = bram_a_we_any ? emb_wr_addr
                                                       : emb_axi_rd_addr;

    wire [8*EMB_DIM-1:0] bram_a_dout [0:N_LANES-1];

    genvar gi;
    generate
        for (gi = 0; gi < N_LANES; gi = gi + 1) begin : gen_bram
            obj_emb_bram #(
                .EMB_DIM    (EMB_DIM),
                .MAX_OBJECTS(MAX_OBJECTS)
            ) u_bram (
                .clk    (clk),
                // Port A – broadcast write + AXI readback
                .a_we   (emb_wr_we),
                .a_addr (bram_a_addr),
                .a_din  (emb_wr_din),
                .a_dout (bram_a_dout[gi]),
                // Port B – per-lane compute read
                .b_addr (lane_bram_addr[gi]),
                .b_dout (lane_bram_dout[gi])
            );
        end
    endgenerate

    // AXI readback uses replica 0 Port-A output
    assign emb_axi_rd_dout = bram_a_dout[0];

    // =========================================================
    //  Scoring lane instances
    // =========================================================
    generate
        for (gi = 0; gi < N_LANES; gi = gi + 1) begin : gen_lane
            scoring_lane #(.EMB_DIM(EMB_DIM)) u_lane (
                .clk            (clk),
                .rst            (rst),
                .start          (lane_fire),
                .task_emb_flat  (task_emb_flat),
                .label_emb_flat (lane_emb[gi]),
                .confidence     (lane_conf[gi]),
                .boost          (lane_boost[gi]),
                .penalty        (lane_penalty[gi]),
                .final_score    (lane_score[gi]),
                .done           (lane_done[gi])
            );
        end
    endgenerate

    // =========================================================
    //  Combinational reduction: max across active lanes
    // =========================================================
    reg signed [31:0] batch_max_score;
    reg [15:0]        batch_max_obj;
    integer           ri;

    always @(*) begin
        batch_max_score = -32'sd2147483648;   // INT_MIN
        batch_max_obj   = 16'h0;
        for (ri = 0; ri < N_LANES; ri = ri + 1) begin
            if (ri[LANE_W-1:0] < n_active) begin
                if ($signed(lane_score[ri]) >= $signed(batch_max_score)) begin
                    batch_max_score = lane_score[ri];
                    batch_max_obj   = lane_obj_id[ri];
                end
            end
        end
    end

    // =========================================================
    //  Dispatch FSM
    // =========================================================
    integer fi;

    always @(posedge clk) begin
        if (rst) begin
            ps_state       <= PS_IDLE;
            done           <= 1'b0;
            best_object    <= 16'h0;
            best_score     <= 32'sd0;
            run_best_score <= -32'sd2147483648;
            run_best_obj   <= 16'h0;
            n_obj_latched  <= 16'h0;
            batch_base     <= 16'h0;
            n_active       <= {LANE_W{1'b0}};
            lane_fire      <= 1'b0;
            for (fi = 0; fi < N_LANES; fi = fi + 1) begin
                lane_bram_addr[fi] <= {OBJ_IDX_W{1'b0}};
                lane_emb[fi]       <= {(8*EMB_DIM){1'b0}};
                lane_conf[fi]      <= 16'sd0;
                lane_boost[fi]     <= 16'sd0;
                lane_penalty[fi]   <= 1'b0;
                lane_obj_id[fi]    <= 16'h0;
            end
        end else begin
            lane_fire <= 1'b0;   // default de-assert

            case (ps_state)

                // ─── Idle: wait for start pulse ──────────────
                PS_IDLE: begin
                    done <= 1'b0;
                    if (start)
                        ps_state <= PS_SETUP;
                end

                // ─── Setup: latch parameters, clear best ─────
                PS_SETUP: begin
                    n_obj_latched  <= n_objects;
                    batch_base     <= 16'h0;
                    run_best_score <= -32'sd2147483648;
                    run_best_obj   <= 16'h0;

                    if (n_objects == 16'h0) begin
                        ps_state <= PS_DONE;
                    end else begin
                        n_active <= (n_objects >= N_LANES)
                                    ? N_LANES[LANE_W-1:0]
                                    : n_objects[LANE_W-1:0];
                        ps_state <= PS_LOAD;
                    end
                end

                // ─── Load: set BRAM addresses for all lanes ──
                PS_LOAD: begin
                    for (fi = 0; fi < N_LANES; fi = fi + 1) begin
                        lane_bram_addr[fi] <= batch_base[OBJ_IDX_W-1:0]
                                              + fi[OBJ_IDX_W-1:0];
                    end
                    ps_state <= PS_PIPE;
                end

                // ─── Pipe: wait 1 clk for BRAM registered out ─
                PS_PIPE: begin
                    ps_state <= PS_LATCH;
                end

                // ─── Latch: capture BRAM + metadata into lanes ─
                PS_LATCH: begin
                    for (fi = 0; fi < N_LANES; fi = fi + 1) begin
                        if (fi[LANE_W-1:0] < n_active) begin
                            lane_emb[fi]     <= lane_bram_dout[fi];
                            lane_conf[fi]    <= conf_mem   [batch_base + fi];
                            lane_boost[fi]   <= boost_mem  [batch_base + fi];
                            lane_penalty[fi] <= penalty_mem[batch_base + fi];
                            lane_obj_id[fi]  <= batch_base + fi[15:0];
                        end
                    end
                    ps_state <= PS_FIRE;
                end

                // ─── Fire: broadcast start to all lanes ──────
                PS_FIRE: begin
                    lane_fire <= 1'b1;
                    ps_state  <= PS_WAIT;
                end

                // ─── Wait: lanes compute in parallel (≈4 cyc) ─
                PS_WAIT: begin
                    if (lane_done[0])
                        ps_state <= PS_REDUCE;
                end

                // ─── Reduce: update running best ─────────────
                PS_REDUCE: begin
                    if ($signed(batch_max_score) >= $signed(run_best_score)) begin
                        run_best_score <= batch_max_score;
                        run_best_obj   <= batch_max_obj;
                    end

                    // More objects left?
                    if (batch_base + n_active >= n_obj_latched) begin
                        ps_state <= PS_DONE;
                    end else begin
                        batch_base <= batch_base + n_active;
                        // n_active for next batch
                        if (n_obj_latched - batch_base - n_active >= N_LANES)
                            n_active <= N_LANES[LANE_W-1:0];
                        else
                            n_active <= n_obj_latched[LANE_W-1:0]
                                        - batch_base[LANE_W-1:0]
                                        - n_active;
                        ps_state <= PS_LOAD;
                    end
                end

                // ─── Done: latch results, signal CPU ─────────
                PS_DONE: begin
                    best_score  <= run_best_score;
                    best_object <= run_best_obj;
                    done        <= 1'b1;
                    ps_state    <= PS_IDLE;
                end

                default: ps_state <= PS_IDLE;
            endcase
        end
    end

    // ── Debug display ────────────────────────────────────────
    `ifdef SIMULATION
    always @(posedge clk) begin
        if (done)
            $display("[SCORER] DONE  best_obj=%0d  best_score=%0d",
                     best_object, best_score);
    end
    `endif

endmodule
