`timescale 1ns / 1ps
/**
 * =============================================================================
 * affinity_scorer_tb.v - Testbench for affinity_scorer_slave_lite_v1_0_S00_AXI
 * =============================================================================
 * DVCon India 2026 - Stage 2B
 * Task-Aware Object Detection - Affinity Scorer Accelerator
 *
 * Description:
 *   Drives the AXI4-Lite slave interface directly (simplified single-beat
 *   write/read tasks, no burst support needed since this is AXI-Lite) to
 *   verify the full accelerator: embedding loading, FSM sequencing, MAC,
 *   score, and max_score, against a known golden reference.
 *
 * Test Scenario:
 *   1 task embedding, 2 detected objects ("wine glass" and "cup") for the
 *   task "Serve wine" - mirrors a real case from the Python pipeline.
 *   Values below are placeholders; REPLACE with actual fixed-point values
 *   exported from your Python score.py / encode.py for a true golden-
 *   reference comparison (see notes at bottom of file).
 *
 * Register Map (must match affinity_scorer_slave_lite_v1_0_S00_AXI.v):
 *   0x00 CTRL          bit0=start, bit1=clr_task, bit2=clr_label
 *   0x04 EMB_WDATA      [7:0] shared serial embedding write port
 *   0x08 SEL            bit0: 0=task embedding, 1=label/object embedding
 *   0x0C N_OBJECTS      [15:0]
 *   0x10 CONFIDENCE     [15:0] signed Q2.14
 *   0x14 BOOST_PENALTY  [15:0]=boost (signed Q2.14), [16]=penalty flag
 *   0x18 RESULT_OBJ     best_object (read-only)
 *   0x1C RESULT_SCORE   best_score (read-only)
 *
 * Tool : Vivado 2025.1 (XSIM) or Questa
 * =============================================================================
 */

module tb;

    localparam EMB_DIM            = 8;
    localparam C_S_AXI_DATA_WIDTH = 32;
    localparam C_S_AXI_ADDR_WIDTH = 5;

    // ─── Clock / Reset ────────────────────────────────────────────────────
    reg clk;
    reg aresetn;

    always #5 clk = ~clk;  // 100 MHz clock (10 ns period)

    // ─── AXI4-Lite Signals (master side, driven by this testbench) ─────────
    reg  [C_S_AXI_ADDR_WIDTH-1:0]   awaddr;
    reg                              awvalid;
    wire                             awready;

    reg  [C_S_AXI_DATA_WIDTH-1:0]   wdata;
    reg  [(C_S_AXI_DATA_WIDTH/8)-1:0] wstrb;
    reg                              wvalid;
    wire                             wready;

    wire [1:0]                       bresp;
    wire                             bvalid;
    reg                              bready;

    reg  [C_S_AXI_ADDR_WIDTH-1:0]   araddr;
    reg                              arvalid;
    wire                             arready;

    wire [C_S_AXI_DATA_WIDTH-1:0]   rdata;
    wire [1:0]                       rresp;
    wire                             rvalid;
    reg                              rready;

    // ─── DUT Instantiation ───────────────────────────────────────────────
    // NOTE: Instantiating affinity_scorer.v (the packaged IP top-level),
    // NOT affinity_scorer_slave_lite_v1_0_S00_AXI directly. affinity_scorer.v
    // is the auto-generated outer wrapper from "Create and Package New IP"
    // and is what would actually be used in a Vivado block design / connected
    // to the VEGA processor. Testing through this level verifies the true
    // external interface of the IP, not just an internal sub-module.
    affinity_scorer #(
        .C_S00_AXI_DATA_WIDTH(C_S_AXI_DATA_WIDTH),
        .C_S00_AXI_ADDR_WIDTH(C_S_AXI_ADDR_WIDTH)
    ) dut (
        .s00_axi_aclk   (clk),
        .s00_axi_aresetn(aresetn),

        .s00_axi_awaddr (awaddr),
        .s00_axi_awprot (3'b000),
        .s00_axi_awvalid(awvalid),
        .s00_axi_awready(awready),

        .s00_axi_wdata  (wdata),
        .s00_axi_wstrb  (wstrb),
        .s00_axi_wvalid (wvalid),
        .s00_axi_wready (wready),

        .s00_axi_bresp  (bresp),
        .s00_axi_bvalid (bvalid),
        .s00_axi_bready (bready),

        .s00_axi_araddr (araddr),
        .s00_axi_arprot (3'b000),
        .s00_axi_arvalid(arvalid),
        .s00_axi_arready(arready),

        .s00_axi_rdata  (rdata),
        .s00_axi_rresp  (rresp),
        .s00_axi_rvalid (rvalid),
        .s00_axi_rready (rready)
    );

    // ─── Golden Reference Test Vectors ──────────────────────────────────────
    // PLACEHOLDER VALUES. Replace these with real exported values from your
    // Python pipeline (see "Exporting Golden Reference" note at end of file).
    //
    // Using small synthetic vectors here ONLY for illustration of the
    // *mechanism* - for a real submission, all 512 elements per vector must
    // be populated from actual CLIP embeddings.

    reg signed [7:0] task_emb   [0:EMB_DIM-1];   // "Serve wine" task embedding
    reg signed [7:0] obj_emb_0  [0:EMB_DIM-1];   // "wine glass" embedding
    reg signed [7:0] obj_emb_1  [0:EMB_DIM-1];   // "cup" embedding

    integer i;

    initial begin
        // Fill with a simple deterministic pattern for now.
        // REPLACE with $readmemh loading real exported hex files, e.g.:
        //   $readmemh("task_emb_serve_wine.hex", task_emb);
        //   $readmemh("obj_emb_wineglass.hex",   obj_emb_0);
        //   $readmemh("obj_emb_cup.hex",         obj_emb_1);
        for (i = 0; i < EMB_DIM; i = i + 1) begin
            task_emb[i]  = (i % 7)  - 3;   // arbitrary small values, range -3..3
            obj_emb_0[i] = (i % 5)  - 2;   // designed to have HIGH similarity
                                            // with task_emb in this synthetic case
            obj_emb_1[i] = (i % 11) - 5;   // designed to have LOWER similarity
        end
    end

    // Expected values - PLACEHOLDER. Replace with real golden reference
    // numbers computed by score.py for these exact embeddings.
    localparam signed [15:0] CONF_OBJ0    = 16'sd13107;  // 0.80 in Q2.14
    localparam signed [15:0] BOOST_OBJ0   = 16'sd4096;   // 0.25 in Q2.14 (preferred)
    localparam                PENALTY_OBJ0 = 1'b0;

    localparam signed [15:0] CONF_OBJ1    = 16'sd11796;  // 0.72 in Q2.14
    localparam signed [15:0] BOOST_OBJ1   = 16'sd1638;   // 0.10 in Q2.14 (keyword)
    localparam                PENALTY_OBJ1 = 1'b0;

    // EXPECTED winner - fill in after running score.py on the same inputs
    localparam [15:0] EXPECTED_BEST_OBJECT = 16'd0;  // object 0 expected to win

    // ─── AXI-Lite Write Task (single beat, full strobe) ─────────────────────
    task axi_write(input [C_S_AXI_ADDR_WIDTH-1:0] addr,
                    input [C_S_AXI_DATA_WIDTH-1:0] data);
        begin
            @(posedge clk);
            awaddr  = addr;
            awvalid = 1'b1;
            wdata   = data;
            wstrb   = 4'hF;
            wvalid  = 1'b1;
            bready  = 1'b1;

            // Wait for the slave to accept both address and data
            wait (awready && wready);
            @(posedge clk);
            awvalid = 1'b0;
            wvalid  = 1'b0;

            // Wait for write response
            wait (bvalid);
            @(posedge clk);
            bready = 1'b0;
        end
    endtask

    // ─── AXI-Lite Read Task (single beat) ────────────────────────────────────
    task axi_read(input  [C_S_AXI_ADDR_WIDTH-1:0] addr,
                  output [C_S_AXI_DATA_WIDTH-1:0] data);
        begin
            @(posedge clk);
            araddr  = addr;
            arvalid = 1'b1;
            rready  = 1'b1;

            wait (arready);
            @(posedge clk);
            arvalid = 1'b0;

            wait (rvalid);
            data = rdata;
            @(posedge clk);
            rready = 1'b0;
        end
    endtask

    // ─── Helper: stream a full 512-element embedding into EMB_WDATA ─────────
    task load_embedding(input is_label,           // 0=task, 1=label/object
                         input signed [7:0] vec [0:EMB_DIM-1]);
        integer k;
        begin
            // Select target buffer
            axi_write(5'h08, {31'd0, is_label});

            // Clear the appropriate loader's write pointer
            if (is_label)
                axi_write(5'h00, 32'h00000004); // bit2 = clr_label
            else
                axi_write(5'h00, 32'h00000002); // bit1 = clr_task

            // Stream all 512 elements
            for (k = 0; k < EMB_DIM; k = k + 1) begin
                axi_write(5'h04, {24'd0, vec[k]});
            end
        end
    endtask

    // ─── Helper: load metadata (confidence/boost/penalty) for one object ────
    task load_metadata(input signed [15:0] confidence,
                        input signed [15:0] boost,
                        input               penalty);
        begin
            axi_write(5'h10, {16'd0, confidence});
            axi_write(5'h14, {15'd0, penalty, boost});
        end
    endtask

    // ─── Main Test Sequence ──────────────────────────────────────────────────
    reg [31:0] readback_obj;
    reg [31:0] readback_score;
    integer    timeout_cycles;

    initial begin
        clk      = 0;
        aresetn  = 0;
        awaddr   = 0; awvalid = 0;
        wdata    = 0; wstrb   = 0; wvalid = 0;
        bready   = 0;
        araddr   = 0; arvalid = 0;
        rready   = 0;

        // ── Reset ──────────────────────────────────────────────────────────
        repeat (5) @(posedge clk);
        aresetn = 1;
        repeat (5) @(posedge clk);

        $display("=========================================================");
        $display(" DVCon India 2026 - Stage 2B - Affinity Scorer Testbench");
        $display("=========================================================");

        // ── Step 1: Load task embedding ("Serve wine") ───────────────────────
        $display("[TB] Loading task embedding...");
        load_embedding(1'b0, task_emb);

        // ── Step 2: Configure n_objects ───────────────────────────────────────
        $display("[TB] Setting n_objects = 2");
        axi_write(5'h0C, 32'd1); //2

        // ── Step 3: Load object 0 ("wine glass") embedding + metadata ────────
        $display("[TB] Loading object 0 (wine glass) embedding...");
        load_embedding(1'b1, obj_emb_0);
        load_metadata(CONF_OBJ0, BOOST_OBJ0, PENALTY_OBJ0);

        // ── Step 4: Start the accelerator ─────────────────────────────────────
        $display("[TB] Asserting start...");
        axi_write(5'h00, 32'h00000001); // bit0 = start

        // ── Step 5: Wait briefly, then load object 1 ("cup") ──────────────────
        // NOTE: In this simplified flow, object 1's embedding/metadata is
        // loaded shortly after start. A more rigorous testbench would poll
        // a "ready_for_next_object" status bit before proceeding -- add this
        // once that handshake signal is wired through to a readable register.
        /*repeat (50) @(posedge clk);

        $display("[TB] Loading object 1 (cup) embedding...");
        load_embedding(1'b1, obj_emb_1);
        load_metadata(CONF_OBJ1, BOOST_OBJ1, PENALTY_OBJ1);*/

        // ── Step 6: Wait for done (with timeout safeguard) ────────────────────
        $display("[TB] Waiting for done...");
        timeout_cycles = 0;
        while (dut.affinity_scorer_slave_lite_v1_0_S00_AXI_inst.done != 1'b1 && timeout_cycles < 5000) begin
            @(posedge clk);
            timeout_cycles = timeout_cycles + 1;
        end

        if (timeout_cycles >= 5000) begin
            $display("[TB] *** TIMEOUT *** done never asserted. Check FSM.");
        end else begin
            $display("[TB] done asserted after %0d cycles", timeout_cycles);
        end

        repeat (5) @(posedge clk);

        // ── Step 7: Read back results ──────────────────────────────────────────
        axi_read(5'h18, readback_obj);
        axi_read(5'h1C, readback_score);

        $display("---------------------------------------------------------");
        $display(" RESULTS");
        $display("---------------------------------------------------------");
        $display(" best_object = %0d", readback_obj);
        $display(" best_score  = %0d", $signed(readback_score));
        $display("---------------------------------------------------------");

        if (readback_obj == EXPECTED_BEST_OBJECT)
            $display(" [PASS] best_object matches expected value");
        else
            $display(" [FAIL] best_object = %0d, expected %0d",
                      readback_obj, EXPECTED_BEST_OBJECT);

        $display("=========================================================");
        $finish;
    end

    // ─── Waveform Dump ────────────────────────────────────────────────────
    initial begin
        $dumpfile("affinity_scorer_tb.vcd");
        $dumpvars(0,tb);
    end

endmodule

/*
 * =============================================================================
 * Exporting Golden Reference Values From Python (for real submission)
 * =============================================================================
 * Add this to score.py (or a standalone script) to export real fixed-point
 * test vectors and expected results, replacing the synthetic placeholders
 * above:
 *
 *   import numpy as np
 *
 *   def to_fixed_q2_14(x):
 *       return int(round(x * (1 << 14)))
 *
 *   def export_embedding_hex(vec, filename):
 *       # vec: numpy array of floats in roughly [-1, 1] range (CLIP output)
 *       # exports as 8-bit signed hex, one value per line, for $readmemh
 *       q = np.clip(np.round(vec * 64), -128, 127).astype(np.int8)
 *       with open(filename, "w") as f:
 *           for v in q:
 *               f.write(f"{v & 0xFF:02x}\n")
 *
 *   # Example usage with your existing pipeline objects:
 *   #   export_embedding_hex(task_embs["Serve wine"], "task_emb_serve_wine.hex")
 *   #   export_embedding_hex(label_embs["wine glass"], "obj_emb_wineglass.hex")
 *   #   export_embedding_hex(label_embs["cup"],        "obj_emb_cup.hex")
 *
 * Then in the testbench, replace the synthetic fill loop with:
 *   $readmemh("task_emb_serve_wine.hex", task_emb);
 *   $readmemh("obj_emb_wineglass.hex",   obj_emb_0);
 *   $readmemh("obj_emb_cup.hex",         obj_emb_1);
 *
 * And compute CONF_OBJ0/BOOST_OBJ0/etc. using to_fixed_q2_14() on the actual
 * confidence/boost values from your score.py run for the same scenario, so
 * RTL output can be directly compared against the real Python final_score
 * and winning object index.
 * =============================================================================
 */
