`timescale 1ns / 1ps
// =============================================================
//  parallel_scorer_tb.v – Core-level Testbench
//  DVCon India 2026
//
//  Tests the parallel_scorer directly (no AXI).
//  Loads embeddings and metadata via write interfaces, fires
//  the scorer, and checks best_object / best_score.
//
//  Test 1: 2 objects (regression – matches Stage 2A golden ref)
//  Test 2: 8 objects (multi-batch with N_LANES=4)
// =============================================================
module parallel_scorer_tb;

    // ── Parameters ──────────────────────────────────────────
    parameter EMB_DIM     = 8;
    parameter MAX_OBJECTS = 16;
    parameter N_LANES     = 4;

    localparam OBJ_IDX_W = $clog2(MAX_OBJECTS);

    // ── Clock / Reset ───────────────────────────────────────
    reg clk;
    reg rst;
    initial clk = 0;
    always #5 clk = ~clk;   // 100 MHz

    // ── DUT signals ─────────────────────────────────────────
    reg         start;
    reg  [15:0] n_objects;
    reg  signed [8*EMB_DIM-1:0] task_emb_flat;

    reg  [EMB_DIM-1:0]           emb_wr_we;
    reg  [OBJ_IDX_W-1:0]        emb_wr_addr;
    reg  [8*EMB_DIM-1:0]        emb_wr_din;

    reg  [OBJ_IDX_W-1:0]        emb_axi_rd_addr;
    wire [8*EMB_DIM-1:0]        emb_axi_rd_dout;

    reg                          meta_conf_we;
    reg                          meta_boost_we;
    reg                          meta_penalty_we;
    reg  [OBJ_IDX_W-1:0]        meta_wr_addr;
    reg  signed [15:0]           meta_wr_conf;
    reg  signed [15:0]           meta_wr_boost;
    reg                          meta_wr_penalty;

    reg  [OBJ_IDX_W-1:0]        meta_rd_addr;
    wire signed [15:0]           meta_rd_conf;
    wire signed [15:0]           meta_rd_boost;
    wire                         meta_rd_penalty;

    wire        done;
    wire [15:0] best_object;
    wire signed [31:0] best_score;

    // ── DUT ─────────────────────────────────────────────────
    parallel_scorer #(
        .EMB_DIM    (EMB_DIM),
        .MAX_OBJECTS(MAX_OBJECTS),
        .N_LANES    (N_LANES)
    ) dut (
        .clk              (clk),
        .rst              (rst),
        .start            (start),
        .n_objects         (n_objects),
        .task_emb_flat     (task_emb_flat),

        .emb_wr_we         (emb_wr_we),
        .emb_wr_addr       (emb_wr_addr),
        .emb_wr_din        (emb_wr_din),

        .emb_axi_rd_addr   (emb_axi_rd_addr),
        .emb_axi_rd_dout   (emb_axi_rd_dout),

        .meta_conf_we      (meta_conf_we),
        .meta_boost_we     (meta_boost_we),
        .meta_penalty_we   (meta_penalty_we),
        .meta_wr_addr      (meta_wr_addr),
        .meta_wr_conf      (meta_wr_conf),
        .meta_wr_boost     (meta_wr_boost),
        .meta_wr_penalty   (meta_wr_penalty),

        .meta_rd_addr      (meta_rd_addr),
        .meta_rd_conf      (meta_rd_conf),
        .meta_rd_boost     (meta_rd_boost),
        .meta_rd_penalty   (meta_rd_penalty),

        .done              (done),
        .best_object       (best_object),
        .best_score        (best_score)
    );

    // ── Waveform ────────────────────────────────────────────
    initial begin
        $dumpfile("parallel_scorer_tb.vcd");
        $dumpvars(0, parallel_scorer_tb);
    end

    // ── Embedding buffers (loaded from hex) ─────────────────
    reg [7:0] task_mem  [0:EMB_DIM-1];
    reg [7:0] obj_mem0  [0:EMB_DIM-1];
    reg [7:0] obj_mem1  [0:EMB_DIM-1];

    integer i, k;
    integer pass_cnt, fail_cnt;

    // ── Tasks ───────────────────────────────────────────────
    // Write one embedding byte to all BRAM replicas
    task write_emb_byte;
        input [OBJ_IDX_W-1:0] obj;
        input integer          byte_idx;
        input [7:0]            data;
        begin
            @(posedge clk); #1;
            emb_wr_addr = obj;
            emb_wr_we   = ({{(EMB_DIM-1){1'b0}}, 1'b1} << byte_idx);
            emb_wr_din  = {EMB_DIM{data}};
            @(posedge clk); #1;
            emb_wr_we = {EMB_DIM{1'b0}};
        end
    endtask

    // Write full embedding for an object
    task write_embedding;
        input [OBJ_IDX_W-1:0] obj;
        input [8*EMB_DIM-1:0]  emb_flat;
        integer b;
        begin
            for (b = 0; b < EMB_DIM; b = b + 1) begin
                write_emb_byte(obj, b, emb_flat[b*8 +: 8]);
            end
        end
    endtask

    // Write metadata for one object
    task write_metadata;
        input [OBJ_IDX_W-1:0] obj;
        input signed [15:0]    conf;
        input signed [15:0]    boost_val;
        input                  pen;
        begin
            @(posedge clk); #1;
            meta_wr_addr = obj;
            meta_wr_conf = conf;
            meta_conf_we = 1'b1;
            @(posedge clk); #1;
            meta_conf_we = 1'b0;

            meta_wr_boost = boost_val;
            meta_boost_we = 1'b1;
            @(posedge clk); #1;
            meta_boost_we = 1'b0;

            meta_wr_penalty = pen;
            meta_penalty_we = 1'b1;
            @(posedge clk); #1;
            meta_penalty_we = 1'b0;
        end
    endtask

    // Flatten a byte array into a packed vector
    function [8*EMB_DIM-1:0] flatten_mem;
        input integer dummy;   // Verilog-2001 requires at least one input
        integer f;
        begin
            flatten_mem = {(8*EMB_DIM){1'b0}};
            // reads from obj_mem0 by default, caller must set up data
        end
    endfunction

    // ── Main Test ───────────────────────────────────────────
    reg [8*EMB_DIM-1:0] task_flat, obj0_flat, obj1_flat;

    initial begin
        $display("==========================================");
        $display("  Parallel Scorer Core Testbench");
        $display("==========================================");

        // Load hex files
        $readmemh("task_emb.hex",  task_mem);
        $readmemh("obj_emb_0.hex", obj_mem0);
        $readmemh("obj_emb_1.hex", obj_mem1);

        // Flatten task embedding
        for (i = 0; i < EMB_DIM; i = i + 1)
            task_flat[i*8 +: 8] = task_mem[i];

        // Flatten object embeddings
        for (i = 0; i < EMB_DIM; i = i + 1) begin
            obj0_flat[i*8 +: 8] = obj_mem0[i];
            obj1_flat[i*8 +: 8] = obj_mem1[i];
        end

        // ── Initialise ──────────────────────────────────────
        start       = 0;
        n_objects   = 0;
        task_emb_flat = {(8*EMB_DIM){1'b0}};
        emb_wr_we   = {EMB_DIM{1'b0}};
        emb_wr_addr = {OBJ_IDX_W{1'b0}};
        emb_wr_din  = {(8*EMB_DIM){1'b0}};
        emb_axi_rd_addr = {OBJ_IDX_W{1'b0}};
        meta_conf_we    = 0;
        meta_boost_we   = 0;
        meta_penalty_we = 0;
        meta_wr_addr    = {OBJ_IDX_W{1'b0}};
        meta_wr_conf    = 0;
        meta_wr_boost   = 0;
        meta_wr_penalty = 0;
        meta_rd_addr    = {OBJ_IDX_W{1'b0}};
        pass_cnt = 0;
        fail_cnt = 0;

        // ── Reset ───────────────────────────────────────────
        rst = 1;
        repeat(10) @(posedge clk);
        rst = 0;
        repeat(5) @(posedge clk);

        // ────────────────────────────────────────────────────
        //  TEST 1: 2 objects (golden reference regression)
        // ────────────────────────────────────────────────────
        $display("\n[TEST 1] 2 objects – golden reference");

        // Load task embedding
        task_emb_flat = task_flat;

        // Load object 0
        write_embedding(0, obj0_flat);
        write_metadata(0, 16'sd13107, 16'sd4096, 1'b0);

        // Load object 1
        write_embedding(1, obj1_flat);
        write_metadata(1, 16'sd11796, 16'sd1638, 1'b0);

        // Configure and start
        n_objects = 16'd2;
        repeat(2) @(posedge clk);

        @(posedge clk); #1;
        start = 1;
        @(posedge clk); #1;
        start = 0;

        // Wait for done
        wait(done);
        repeat(2) @(posedge clk);

        // Check results
        $display("  best_object = %0d  (expected 0)", best_object);
        $display("  best_score  = %0d  (expected 11749)", best_score);

        if (best_object == 16'd0) begin
            $display("  [PASS] best_object");
            pass_cnt = pass_cnt + 1;
        end else begin
            $display("  [FAIL] best_object");
            fail_cnt = fail_cnt + 1;
        end

        if (best_score == 32'sd11749) begin
            $display("  [PASS] best_score");
            pass_cnt = pass_cnt + 1;
        end else begin
            $display("  [FAIL] best_score");
            fail_cnt = fail_cnt + 1;
        end

        repeat(5) @(posedge clk);

        // ────────────────────────────────────────────────────
        //  TEST 2: 8 objects (exercises 2 batches with N_LANES=4)
        // ────────────────────────────────────────────────────
        $display("\n[TEST 2] 8 objects – multi-batch");

        // Objects 0-3: copies of obj0 with varying metadata
        // Objects 4-7: copies of obj1 with varying metadata
        // Object 5 should win: obj1 embedding + highest boost
        for (k = 0; k < 4; k = k + 1) begin
            write_embedding(k, obj0_flat);
            write_metadata(k, 16'sd8000, 16'sd1000, 1'b0);
        end
        for (k = 4; k < 8; k = k + 1) begin
            write_embedding(k, obj1_flat);
            write_metadata(k, 16'sd8000, 16'sd1000, 1'b0);
        end
        // Give object 5 the highest boost
        write_metadata(5, 16'sd13107, 16'sd4096, 1'b0);

        n_objects = 16'd8;
        repeat(2) @(posedge clk);

        @(posedge clk); #1;
        start = 1;
        @(posedge clk); #1;
        start = 0;

        wait(done);
        repeat(2) @(posedge clk);

        $display("  best_object = %0d  (expected 5)", best_object);
        $display("  best_score  = %0d", best_score);

        if (best_object == 16'd5) begin
            $display("  [PASS] best_object");
            pass_cnt = pass_cnt + 1;
        end else begin
            $display("  [FAIL] best_object (got %0d)", best_object);
            fail_cnt = fail_cnt + 1;
        end

        // ── Summary ─────────────────────────────────────────
        $display("\n==========================================");
        $display("  PASS: %0d    FAIL: %0d", pass_cnt, fail_cnt);
        if (fail_cnt == 0)
            $display("  *** ALL TESTS PASSED ***");
        else
            $display("  *** FAILURES DETECTED ***");
        $display("==========================================");

        repeat(10) @(posedge clk);
        $finish;
    end

    // ── Watchdog ────────────────────────────────────────────
    initial begin
        #100_000;
        $display("[WATCHDOG] Timeout reached");
        $finish;
    end

endmodule
