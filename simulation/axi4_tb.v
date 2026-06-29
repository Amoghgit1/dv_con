`timescale 1ns / 1ps
// =============================================================
//  axi4_tb.v  -  AXI4-Full Testbench (Parallel BRAM Scorer)
//  DVCon India 2026
//
//  Tests the BRAM-backed parallel scorer via AXI4-Full.
//  Test 1: Register write/readback (N_OBJECTS)
//  Test 2: 2-object golden reference (regression)
//  Test 3: 8-object multi-batch (exercises N_LANES=4)
// =============================================================
module axi4_tb;

    // ── Parameters ─────────────────────────────────────────────
    parameter EMB_DIM     = 8;
    parameter MAX_OBJECTS = 16;
    parameter N_LANES     = 4;
    parameter CLK_HALF    = 5;

    // ── AXI master signals ──────────────────────────────────────
    reg         ACLK;
    reg         ARESETn;

    reg  [3:0]  AWID;
    reg  [31:0] AWADDR;
    reg  [7:0]  AWLEN;
    reg  [2:0]  AWSIZE;
    reg  [1:0]  AWBURST;
    reg         AWVALID;
    wire        AWREADY;

    reg  [31:0] WDATA;
    reg  [3:0]  WSTRB;
    reg         WLAST;
    reg         WVALID;
    wire        WREADY;

    wire [3:0]  BID;
    wire [1:0]  BRESP;
    wire        BVALID;
    reg         BREADY;

    reg  [3:0]  ARID;
    reg  [31:0] ARADDR;
    reg  [7:0]  ARLEN;
    reg  [2:0]  ARSIZE;
    reg  [1:0]  ARBURST;
    reg         ARVALID;
    wire        ARREADY;

    wire [3:0]  RID;
    wire [31:0] RDATA;
    wire [1:0]  RRESP;
    wire        RLAST;
    wire        RVALID;
    reg         RREADY;

    // ── DUT ────────────────────────────────────────────────────
    axi4_affinity_top #(
        .EMB_DIM    (EMB_DIM),
        .MAX_OBJECTS(MAX_OBJECTS),
        .N_LANES    (N_LANES)
    ) dut (
        .ACLK   (ACLK),    .ARESETn(ARESETn),
        .AWID   (AWID),     .AWADDR (AWADDR),  .AWLEN  (AWLEN),
        .AWSIZE (AWSIZE),   .AWBURST(AWBURST), .AWVALID(AWVALID),
        .AWREADY(AWREADY),
        .WDATA  (WDATA),    .WSTRB  (WSTRB),   .WLAST  (WLAST),
        .WVALID (WVALID),   .WREADY (WREADY),
        .BID    (BID),      .BRESP  (BRESP),   .BVALID (BVALID),
        .BREADY (BREADY),
        .ARID   (ARID),     .ARADDR (ARADDR),  .ARLEN  (ARLEN),
        .ARSIZE (ARSIZE),   .ARBURST(ARBURST), .ARVALID(ARVALID),
        .ARREADY(ARREADY),
        .RID    (RID),      .RDATA  (RDATA),   .RRESP  (RRESP),
        .RLAST  (RLAST),    .RVALID (RVALID),  .RREADY (RREADY)
    );

    // ── Clock ───────────────────────────────────────────────────
    initial ACLK = 0;
    always  #CLK_HALF ACLK = ~ACLK;

    // ── Waveform ────────────────────────────────────────────────
    initial begin
        $dumpfile("axi4_affinity_tb.vcd");
        $dumpvars(0, axi4_tb);
    end

    // ── Embedding buffers ───────────────────────────────────────
    reg [7:0]  task_mem [0:EMB_DIM-1];
    reg [7:0]  obj_mem0 [0:EMB_DIM-1];
    reg [7:0]  obj_mem1 [0:EMB_DIM-1];
    reg [7:0]  obj_mem2 [0:EMB_DIM-1];
    reg [7:0]  obj_mem3 [0:EMB_DIM-1];
    reg [7:0]  obj_mem4 [0:EMB_DIM-1];
    reg [7:0]  obj_mem5 [0:EMB_DIM-1];
    reg [7:0]  obj_mem6 [0:EMB_DIM-1];
    reg [7:0]  obj_mem7 [0:EMB_DIM-1];
    reg [31:0] burst_buf[0:EMB_DIM-1];

    integer    i;
    reg [31:0] rd_result;

    // ==========================================================
    //  AXI MASTER TASKS
    // ==========================================================

    // ── Single-beat write ───────────────────────────────────────
    task axi_write;
        input [31:0] addr;
        input [31:0] data;
        begin
            @(posedge ACLK); #1;
            AWVALID = 1; AWADDR = addr; AWLEN = 8'h00;
            AWSIZE  = 3'b010; AWBURST = 2'b01; AWID = 4'h0;

            wait(AWREADY); @(posedge ACLK); #1;
            AWVALID = 0;

            WVALID = 1; WDATA = data; WSTRB = 4'hF; WLAST = 1;
            wait(WREADY); @(posedge ACLK); #1;
            WVALID = 0; WLAST = 0;

            BREADY = 1;
            wait(BVALID); @(posedge ACLK); #1;
            BREADY = 0;
        end
    endtask

    // ── Burst write: reads data from burst_buf[0..n-1] ─────────
    task axi_burst_write;
        input [31:0] base_addr;
        input integer n_beats;
        integer b;
        begin
            @(posedge ACLK); #1;
            AWVALID = 1; AWADDR = base_addr; AWLEN = n_beats - 1;
            AWSIZE  = 3'b010; AWBURST = 2'b01; AWID = 4'h1;

            wait(AWREADY); @(posedge ACLK); #1;
            AWVALID = 0;

            for (b = 0; b < n_beats; b = b + 1) begin
                WVALID = 1;
                WDATA  = burst_buf[b];
                WSTRB  = 4'hF;
                WLAST  = (b == n_beats - 1) ? 1'b1 : 1'b0;
                wait(WREADY); @(posedge ACLK); #1;
            end
            WVALID = 0; WLAST = 0;

            BREADY = 1;
            wait(BVALID); @(posedge ACLK); #1;
            BREADY = 0;
        end
    endtask

    // ── Single-beat read ────────────────────────────────────────
    task axi_read;
        input  [31:0] addr;
        output [31:0] rdata;
        begin
            @(posedge ACLK); #1;
            ARVALID = 1; ARADDR = addr; ARLEN = 8'h00;
            ARSIZE  = 3'b010; ARBURST = 2'b01; ARID = 4'h0;

            wait(ARREADY); @(posedge ACLK); #1;
            ARVALID = 0;

            RREADY = 1;
            wait(RVALID);
            rdata  = RDATA;
            @(posedge ACLK); #1;
            RREADY = 0;
        end
    endtask

    // ── Poll STATUS[DONE] ───────────────────────────────────────
    task wait_done;
        integer timeout;
        begin
            timeout   = 0;
            rd_result = 32'h0;
            while (rd_result[0] == 1'b0) begin
                axi_read(32'h0004, rd_result);
                timeout = timeout + 1;
                if (timeout > 2000) begin
                    $display("[TIMEOUT] Accelerator did not complete");
                    $stop;
                end
            end
            $display("[INFO]  DONE after %0d polls", timeout);
        end
    endtask

    // ── Helper: programme one object fully ──────────────────────
    task programme_object;
        input integer obj_idx;
        input [31:0]  conf;
        input [31:0]  boost_val;
        input [31:0]  penalty_val;
        // Uses burst_buf (caller must fill it with embedding bytes)
        begin
            axi_write(32'h0200 + obj_idx * 4, conf);
            axi_write(32'h0300 + obj_idx * 4, boost_val);
            axi_write(32'h0400 + obj_idx * 4, penalty_val);
            axi_burst_write(32'h1000 + obj_idx * EMB_DIM * 4, EMB_DIM);
        end
    endtask

    // ==========================================================
    //  MAIN TEST
    // ==========================================================
    integer pass_cnt, fail_cnt, k;

    initial begin
        $display("==========================================");
        $display("  AXI4-Full Parallel Scorer Testbench     ");
        $display("==========================================");

        // Initialise bus signals
        AWVALID = 0; WVALID  = 0; BREADY  = 0;
        ARVALID = 0; RREADY  = 0;
        AWID    = 0; AWADDR  = 0; AWLEN   = 0;
        AWSIZE  = 0; AWBURST = 0;
        WDATA   = 0; WSTRB   = 0; WLAST   = 0;
        ARID    = 0; ARADDR  = 0; ARLEN   = 0;
        ARSIZE  = 0; ARBURST = 0;
        ARESETn = 0;
        pass_cnt = 0; fail_cnt = 0;

        // Load hex files
        $readmemh("task_emb.hex",  task_mem);
        $readmemh("obj_emb_0.hex", obj_mem0);
        $readmemh("obj_emb_1.hex", obj_mem1);
        $readmemh("obj_emb_2.hex", obj_mem2);
        $readmemh("obj_emb_3.hex", obj_mem3);
        $readmemh("obj_emb_4.hex", obj_mem4);
        $readmemh("obj_emb_5.hex", obj_mem5);
        $readmemh("obj_emb_6.hex", obj_mem6);
        $readmemh("obj_emb_7.hex", obj_mem7);

        // ── Reset ────────────────────────────────────────────
        repeat(10) @(posedge ACLK);
        ARESETn = 1;
        repeat(5)  @(posedge ACLK);

        // ══════════════════════════════════════════════════════
        //  TEST 1: Register write/readback
        // ══════════════════════════════════════════════════════
        $display("\n[TEST 1] N_OBJECTS write/readback");
        axi_write(32'h0008, 32'd2);
        axi_read (32'h0008, rd_result);
        if (rd_result[15:0] == 16'd2) begin
            $display("  [PASS] N_OBJECTS = %0d", rd_result[15:0]);
            pass_cnt = pass_cnt + 1;
        end else begin
            $display("  [FAIL] expected 2, got %0d", rd_result[15:0]);
            fail_cnt = fail_cnt + 1;
        end

        // ══════════════════════════════════════════════════════
        //  TEST 2: 2-object golden reference
        // ══════════════════════════════════════════════════════
        $display("\n[TEST 2] 2 objects - golden reference");

        // Task embedding burst
        for (i = 0; i < EMB_DIM; i = i + 1)
            burst_buf[i] = {24'h0, task_mem[i]};
        axi_burst_write(32'h0100, EMB_DIM);

        // Object 0
        for (i = 0; i < EMB_DIM; i = i + 1)
            burst_buf[i] = {24'h0, obj_mem0[i]};
        programme_object(0, 32'd13107, 32'd4096, 32'h0);

        // Object 1
        for (i = 0; i < EMB_DIM; i = i + 1)
            burst_buf[i] = {24'h0, obj_mem1[i]};
        programme_object(1, 32'd11796, 32'd1638, 32'h0);

        // Start
        axi_write(32'h0000, 32'h1);
        wait_done();

        // Verify BEST_OBJECT
        axi_read(32'h000C, rd_result);
        $display("  BEST_OBJECT = %0d", rd_result[15:0]);
        if (rd_result[15:0] == 16'd0) begin
            $display("  [PASS]");
            pass_cnt = pass_cnt + 1;
        end else begin
            $display("  [FAIL] expected 0");
            fail_cnt = fail_cnt + 1;
        end

        // Verify BEST_SCORE
        axi_read(32'h0010, rd_result);
        $display("  BEST_SCORE  = %0d", $signed(rd_result));
        if ($signed(rd_result) == 32'sd12627) begin
            $display("  [PASS]");
            pass_cnt = pass_cnt + 1;
        end else begin
            $display("  [FAIL] expected 12627");
            fail_cnt = fail_cnt + 1;
        end

        // ══════════════════════════════════════════════════════
        //  TEST 3: 8 objects - multi-batch
        // ══════════════════════════════════════════════════════
        $display("\n[TEST 3] 8 objects - multi-batch (2 rounds)");

        axi_write(32'h0008, 32'd8);    // N_OBJECTS = 8

        // ---------------- Object 0 ----------------
        for(i=0;i<EMB_DIM;i=i+1)
            burst_buf[i]={24'h0,obj_mem0[i]};
        programme_object(0,32'd13107,32'd4096,32'h0);
        
        // ---------------- Object 1 ----------------
        for(i=0;i<EMB_DIM;i=i+1)
            burst_buf[i]={24'h0,obj_mem1[i]};
        programme_object(1,32'd11796,32'd1638,32'h0);
        
        // ---------------- Object 2 ----------------
        for(i=0;i<EMB_DIM;i=i+1)
            burst_buf[i]={24'h0,obj_mem2[i]};
        programme_object(2,32'd12451,32'd2949,32'h0);
        
        // ---------------- Object 3 ----------------
        for(i=0;i<EMB_DIM;i=i+1)
            burst_buf[i]={24'h0,obj_mem3[i]};
        programme_object(3,32'd9994,32'd819,32'h0);
        
        // ---------------- Object 4 ----------------
        for(i=0;i<EMB_DIM;i=i+1)
            burst_buf[i]={24'h0,obj_mem4[i]};
        programme_object(4,32'd11141,32'd1966,32'h0);
        
        // ---------------- Object 5 ----------------
        for(i=0;i<EMB_DIM;i=i+1)
            burst_buf[i]={24'h0,obj_mem5[i]};
        programme_object(5,32'd14909,32'd4915,32'h0);
        
        // ---------------- Object 6 ----------------
        for(i=0;i<EMB_DIM;i=i+1)
            burst_buf[i]={24'h0,obj_mem6[i]};
        programme_object(6,32'd9502,32'd1310,32'h0);
        
        // ---------------- Object 7 ----------------
        for(i=0;i<EMB_DIM;i=i+1)
            burst_buf[i]={24'h0,obj_mem7[i]};
        programme_object(7,32'd12124,32'd2457,32'h0);

    
        // Start
        axi_write(32'h0000, 32'h1);
        wait_done();

        axi_read(32'h000C, rd_result);
        $display("  BEST_OBJECT = %0d  (expected 5)", rd_result[15:0]);
        if (rd_result[15:0] == 16'd5) begin
            $display("  [PASS]");
            pass_cnt = pass_cnt + 1;
        end else begin
            $display("  [FAIL]");
            fail_cnt = fail_cnt + 1;
        end

        axi_read(32'h0010, rd_result);
        $display("  BEST_SCORE  = %0d", $signed(rd_result));

        // ── Summary ───────────────────────────────────────────
        $display("\n==========================================");
        $display("  PASS: %0d    FAIL: %0d", pass_cnt, fail_cnt);
        if (fail_cnt == 0)
            $display("  *** ALL TESTS PASSED ***");
        else
            $display("  *** FAILURES DETECTED ***");
        $display("==========================================");

        repeat(20) @(posedge ACLK);
        $finish;
    end

    // ── Watchdog ─────────────────────────────────────────────
    initial begin
        #2_000_000;
        $display("[WATCHDOG] 2 ms limit reached");
        $finish;
    end

endmodule