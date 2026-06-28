`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 28.06.2026 20:26:17
// Design Name: 
// Module Name: axi_tb
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////




/**
 * =============================================================================
 * top_tb.v - AXI4 Full Testbench for myip (Affinity Scorer Accelerator)
 * =============================================================================
 * DVCon India 2026 - Stage 2B
 *
 * Exact Register Map (derived from slave BRAM logic):
 *
 *   WRITE (CPU → Accelerator):
 *   0x00 : start      (bit 0, self-clearing via done_reg logic)
 *   0x04 : n_objects  [15:0]  = {WDATA[15:8], WDATA[7:0]}
 *   0x08 : confidence [15:0]  = {WDATA[15:8], WDATA[7:0]}
 *   0x0C : boost[15:0] + penalty = WDATA[31:16] unused,
 *                                   WDATA[15:8]=boost_hi,
 *                                   WDATA[7:0] =boost_lo,
 *                                   penalty    =WDATA[24] (byte3 bit0)
 *   0x10 : task_emb[3:0]  packed = {emb[3], emb[2], emb[1], emb[0]}
 *   0x14 : task_emb[7:4]  packed = {emb[7], emb[6], emb[5], emb[4]}
 *   0x18 : label_emb[3:0] packed = {emb[3], emb[2], emb[1], emb[0]}
 *   0x1C : label_emb[7:4] packed = {emb[7], emb[6], emb[5], emb[4]}
 *
 *   READ (Accelerator → CPU):
 *   0x20 : best_score  [31:0]
 *   0x24 : best_object [15:0]
 *   0x28 : done_reg    [0]
 *
 * Golden Reference (Python, EMB_DIM=8, seed=42):
 *   task_emb  = [13,-4,17,39,-6,-6,41,20]
 *   obj_emb_0 = [-11,12,-11,-11,5,-43,-39,-13]  conf=13107 boost=4096 pen=0
 *   obj_emb_1 = [-23,7,-20,-32,33,-5,2,-32]     conf=11796 boost=1638 pen=0
 *   Expected winner: object 0, best_score = 11749
 *
 * FSM note: n_objects=2, one start pulse. FSM auto-loops through both objects.
 * Both label embeddings must be loaded BEFORE start (the FSM reads label_emb
 * immediately on entering LOAD state - for a 2-object run with shared BRAM,
 * only obj0 embedding is needed at start; obj1 is written while FSM processes obj0.
 *
 * Tool: Vivado 2025.1 XSIM / Questa
 * =============================================================================
 */


module axi_tb();
    localparam EMB_DIM  = 8;
    localparam CLK_HALF = 5;    // 100 MHz
    localparam DATA_W   = 32;
    localparam ADDR_W   = 9;
    localparam ID_W     = 1;

    localparam EXPECTED_BEST_OBJECT = 16'd0;
    localparam EXPECTED_BEST_SCORE  = 32'd11749;

    // ─── Clock & Reset ────────────────────────────────────────────────────
    reg clk;
    reg resetn;
    initial clk = 0;
    always #CLK_HALF clk = ~clk;

    // ─── AXI4 Full Slave Signals ──────────────────────────────────────────
    reg  [ID_W-1:0]     awid;
    reg  [ADDR_W-1:0]   awaddr;
    reg  [7:0]          awlen;
    reg  [2:0]          awsize;
    reg  [1:0]          awburst;
    reg                 awlock, awvalid;
    reg  [3:0]          awcache, awqos, awregion;
    reg  [2:0]          awprot;
    wire                awready;

    reg  [DATA_W-1:0]   wdata;
    reg  [DATA_W/8-1:0] wstrb;
    reg                 wlast, wvalid;
    wire                wready;

    wire [ID_W-1:0]     bid;
    wire [1:0]          bresp;
    wire                bvalid;
    reg                 bready;

    reg  [ID_W-1:0]     arid;
    reg  [ADDR_W-1:0]   araddr;
    reg  [7:0]          arlen;
    reg  [2:0]          arsize;
    reg  [1:0]          arburst;
    reg                 arlock, arvalid;
    reg  [3:0]          arcache, arqos, arregion;
    reg  [2:0]          arprot;
    wire                arready;

    wire [ID_W-1:0]     rid;
    wire [DATA_W-1:0]   rdata;
    wire [1:0]          rresp;
    wire                rlast, rvalid;
    reg                 rready;

    wire                irq;

    // ─── DUT ─────────────────────────────────────────────────────────────
    myip #(.EMB_DIM(EMB_DIM)) dut (
        .s00_axi_aclk    (clk),
        .s00_axi_aresetn (resetn),
        .s00_axi_awid    (awid),
        .s00_axi_awaddr  (awaddr),
        .s00_axi_awlen   (awlen),
        .s00_axi_awsize  (awsize),
        .s00_axi_awburst (awburst),
        .s00_axi_awlock  (awlock),
        .s00_axi_awcache (awcache),
        .s00_axi_awprot  (awprot),
        .s00_axi_awqos   (awqos),
        .s00_axi_awregion(awregion),
        .s00_axi_awuser  (1'b0),
        .s00_axi_awvalid (awvalid),
        .s00_axi_awready (awready),
        .s00_axi_wdata   (wdata),
        .s00_axi_wstrb   (wstrb),
        .s00_axi_wlast   (wlast),
        .s00_axi_wuser   (1'b0),
        .s00_axi_wvalid  (wvalid),
        .s00_axi_wready  (wready),
        .s00_axi_bid     (bid),
        .s00_axi_bresp   (bresp),
        .s00_axi_buser   (),
        .s00_axi_bvalid  (bvalid),
        .s00_axi_bready  (bready),
        .s00_axi_arid    (arid),
        .s00_axi_araddr  (araddr),
        .s00_axi_arlen   (arlen),
        .s00_axi_arsize  (arsize),
        .s00_axi_arburst (arburst),
        .s00_axi_arlock  (arlock),
        .s00_axi_arcache (arcache),
        .s00_axi_arprot  (arprot),
        .s00_axi_arqos   (arqos),
        .s00_axi_arregion(arregion),
        .s00_axi_aruser  (1'b0),
        .s00_axi_arvalid (arvalid),
        .s00_axi_arready (arready),
        .s00_axi_rid     (rid),
        .s00_axi_rdata   (rdata),
        .s00_axi_rresp   (rresp),
        .s00_axi_rlast   (rlast),
        .s00_axi_ruser   (),
        .s00_axi_rvalid  (rvalid),
        .s00_axi_rready  (rready),
        // Master - tied off
        .m00_axi_init_axi_txn(1'b0),
        .m00_axi_txn_done    (),
        .m00_axi_error       (),
        .m00_axi_aclk        (clk),
        .m00_axi_aresetn     (resetn),
        .m00_axi_awready(1'b0), .m00_axi_wready(1'b0),
        .m00_axi_bid(1'b0), .m00_axi_bresp(2'b0),
        .m00_axi_buser(1'b0), .m00_axi_bvalid(1'b0),
        .m00_axi_arready(1'b0), .m00_axi_rid(1'b0),
        .m00_axi_rdata(32'b0), .m00_axi_rresp(2'b0),
        .m00_axi_rlast(1'b0), .m00_axi_ruser(1'b0),
        .m00_axi_rvalid(1'b0),
        // Interrupt - tied off
        .s_axi_intr_aclk(clk), .s_axi_intr_aresetn(resetn),
        .s_axi_intr_awaddr(5'b0), .s_axi_intr_awprot(3'b0),
        .s_axi_intr_awvalid(1'b0), .s_axi_intr_wdata(32'b0),
        .s_axi_intr_wstrb(4'b0), .s_axi_intr_wvalid(1'b0),
        .s_axi_intr_bready(1'b1), .s_axi_intr_araddr(5'b0),
        .s_axi_intr_arprot(3'b0), .s_axi_intr_arvalid(1'b0),
        .s_axi_intr_rready(1'b1), .irq(irq)
    );

    // ─── Waveform Dump ────────────────────────────────────────────────────
    initial begin
        $dumpfile("axi_tb.vcd");
        $dumpvars(0, axi_tb);
    end

    // ─── AXI4 Write Task (separate AW and W, then wait B) ────────────────
    task axi_write;
        input [ADDR_W-1:0]   addr;
        input [DATA_W-1:0]   data;
        input [DATA_W/8-1:0] strb;   // byte enables
        begin
            // AW channel
            @(posedge clk); #1;
            awaddr  = addr;
            awlen   = 8'd0;
            awsize  = 3'b010;
            awburst = 2'b01;
            awid    = 1'b0;
            awlock  = 1'b0; awcache = 4'b0010;
            awprot  = 3'b0; awqos   = 4'b0; awregion = 4'b0;
            awvalid = 1'b1;
            wait (awready === 1'b1);
            @(posedge clk); #1;
            awvalid = 1'b0;

            // W channel
            wdata  = data;
            wstrb  = strb;
            wlast  = 1'b1;
            wvalid = 1'b1;
            wait (wready === 1'b1);
            @(posedge clk); #1;
            wvalid = 1'b0;
            wlast  = 1'b0;

            // B channel
            bready = 1'b1;
            wait (bvalid === 1'b1);
            if (bresp !== 2'b00)
                $display("[TB] WARNING: BRESP=%b at addr=0x%h", bresp, addr);
            @(posedge clk); #1;
        end
    endtask

    // ─── AXI4 Read Task ───────────────────────────────────────────────────
    reg [DATA_W-1:0] rd_result;
    task axi_read;
        input  [ADDR_W-1:0] addr;
        output [DATA_W-1:0] data;
        begin
            @(posedge clk); #1;
            araddr  = addr;
            arlen   = 8'd0;
            arsize  = 3'b010;
            arburst = 2'b01;
            arid    = 1'b0;
            arlock  = 1'b0; arcache = 4'b0010;
            arprot  = 3'b0; arqos   = 4'b0; arregion = 4'b0;
            arvalid = 1'b1;
            rready  = 1'b1;
            wait (arready === 1'b1);
            @(posedge clk); #1;
            arvalid = 1'b0;
            wait (rvalid === 1'b1);
            data = rdata;
            if (rresp !== 2'b00)
                $display("[TB] WARNING: RRESP=%b at addr=0x%h", rresp, addr);
            @(posedge clk); #1;
            rready = 1'b0;
        end
    endtask

    // ─── Test Vectors ─────────────────────────────────────────────────────
    // Golden: task=[13,-4,17,39,-6,-6,41,20]
    //         obj0=[-11,12,-11,-11,5,-43,-39,-13] conf=13107 boost=4096
    //         obj1=[-23,7,-20,-32,33,-5,2,-32]    conf=11796 boost=1638
    //         winner=object0, best_score=11749

    // Packed 32-bit words for embedding writes
    // task_emb[3:0] packed: {emb[3], emb[2], emb[1], emb[0]}
    //   = {8'd39, 8'd17, 8'hFC(-4 as uint8), 8'd13}
    localparam [31:0] TASK_WORD0 = {8'd39, 8'd17, 8'hFC, 8'd13};

    localparam [31:0] TASK_WORD1 = {8'd20, 8'd41, 8'hFA, 8'hFA};
    
    localparam [31:0] OBJ0_WORD0 = {8'hF5, 8'hF5, 8'd12, 8'hF5};
    
    localparam [31:0] OBJ0_WORD1 = {8'hF3, 8'hD9, 8'hD5, 8'd5};
    
    //localparam [31:0] OBJ1_WORD0 = {8'hE0, 8'hEC, 8'd7, 8'hE9};
    
    //localparam [31:0] OBJ1_WORD1 = {8'hE0, 8'd2, 8'hFB, 8'd33};

    // ─── Monitoring ───────────────────────────────────────────────────────
    reg [2:0] prev_fsm_state;
    always @(posedge clk) begin
        if (resetn) begin
            prev_fsm_state <= dut.core.FSM.state;
            if (dut.core.FSM.state !== prev_fsm_state)
                $display("[FSM] t=%0t  %0d→%0d  obj=%0d   mac_en=%b  done=%b",
                    $time,
                    prev_fsm_state, 
                    dut.core.FSM.state,
                    dut.core.FSM.obj_out,
                    //dut.core.emd,
                    dut.core.FSM.mac_en,
                    dut.core.FSM.done);
        end
    end
    always @(posedge clk) begin
    if (dut.core.mac_done)
        $display("MAC DONE @ %0t", $time);

    if (dut.core.score_done)
        $display("SCORE DONE @ %0t", $time);

    if (dut.core.max_done)
        $display("MAX DONE @ %0t", $time);

    if (dut.core.done)
        $display("FSM DONE @ %0t", $time);
end
    // ─── Result regs ──────────────────────────────────────────────────────
    reg [DATA_W-1:0] rb_done, rb_score, rb_object;
    integer          timeout;

    // ─── Main Test ────────────────────────────────────────────────────────
    initial begin
        // Default AXI signal state
        awid=0; awaddr=0; awlen=0; awsize=0; awburst=0;
        awlock=0; awcache=0; awprot=0; awqos=0; awregion=0; awvalid=0;
        wdata=0; wstrb=0; wlast=0; wvalid=0; bready=1;
        arid=0; araddr=0; arlen=0; arsize=0; arburst=0;
        arlock=0; arcache=0; arprot=0; arqos=0; arregion=0;
        arvalid=0; rready=0;

        resetn = 1'b0;
        repeat(10) @(posedge clk);
        resetn = 1'b1;
        repeat(10) @(posedge clk);

        $display("==========================================================");
        $display(" DVCon India 2026 - Stage 2B - AXI4 Full Testbench");
        $display("==========================================================");

        // ── Step 1: Write task embedding ──────────────────────────────────
        // task_emb[3:0] at word 4 (0x10), all 4 byte lanes
        // task_emb[7:4] at word 5 (0x14), all 4 byte lanes
        $display("[TB] Step 1: Writing task embedding...");
        axi_write(9'h010, TASK_WORD0, 4'hF);
        axi_write(9'h014, TASK_WORD1, 4'hF);

        // ── Step 2: Write object 0 label embedding ────────────────────────
        // label_emb[3:0] at word 6 (0x18), label_emb[7:4] at word 7 (0x1C)
        $display("[TB] Step 2: Writing object 0 (wine glass) label embedding...");
        axi_write(9'h018, OBJ0_WORD0, 4'hF);
        axi_write(9'h01C, OBJ0_WORD1, 4'hF);

        // ── Step 3: Write n_objects = 2 ───────────────────────────────────
        // n_objects at word 1 (0x04), bytes 0 and 1
        $display("[TB] Step 3: Writing n_objects=2...");
        axi_write(9'h004, 32'h00000001, 4'h3);  // bytes 0,1 only

        // ── Step 4: Write confidence for object 0 ─────────────────────────
        // confidence at word 2 (0x08), bytes 0 and 1
        // conf0 = 13107 = 0x3333
        $display("[TB] Step 4: Writing confidence=13107...");
        axi_write(9'h008, 32'h00003333, 4'h3);

        // ── Step 5: Write boost + penalty for object 0 ────────────────────
        // boost at word 3 (0x0C), bytes 0 and 1
        // penalty at byte 3 of word 3
        // boost0 = 4096 = 0x1000, penalty = 0
        $display("[TB] Step 5: Writing boost=4096, penalty=0...");
        axi_write(9'h00C, 32'h00001000, 4'hF);

        // ── Step 6: Assert start (word 0, bit 0) ──────────────────────────
        $display("[TB] Step 6: Asserting start...");
        axi_write(9'h000, 32'h00000001, 4'h1);  // byte 0 only
        repeat(2) @(posedge clk);

        axi_write(9'h000, 32'h00000000, 4'h1);
        // Small gap - FSM needs a few cycles to see start and begin LOAD
        repeat(2) @(posedge clk);

        /*// ── Step 7: While FSM processes obj0, load obj1 embedding ─────────
        // FSM is in MAC state (~1 cycle for EMB_DIM=8 combinational MAC)
        // Safe to overwrite label BRAM now since MAC already latched obj0
        $display("[TB] Step 7: Writing object 1 (cup) label embedding...");
        axi_write(9'h018, OBJ1_WORD0, 4'hF);
        axi_write(9'h01C, OBJ1_WORD1, 4'hF);

        // Update confidence and boost for object 1
        // conf1=11796=0x2E14, boost1=1638=0x0666
        $display("[TB] Step 7b: Writing obj1 confidence=11796 boost=1638...");
        axi_write(9'h008, 32'h00002E14, 4'h3);
        axi_write(9'h00C, 32'h00000666, 4'hF);*/

        // ── Step 8: Poll done register (0x28) ─────────────────────────────
        $display("[TB] Step 8: Polling done at 0x028...");
        timeout = 0;
        rb_done = 32'd0;
        while (rb_done[0] !== 1'b1 && timeout < 5000) begin
            repeat(5) @(posedge clk);
            axi_read(9'h028, rb_done);
            timeout = timeout + 1;
        end

        if (timeout >= 5000) begin
            $display("[TB] *** TIMEOUT: done never asserted ***");
            $display("[TB]     Check: FSM reset polarity, emd_signal, done_reg mux");
            $finish;
        end
        $display("[TB] done=1 after ~%0d polls", timeout);

        // ── Step 9: Read results ──────────────────────────────────────────
        $display("[TB] Step 9: Reading best_score (0x020) and best_object (0x024)...");
        axi_read(9'h020, rb_object);
        axi_read(9'h024, rb_score);

        // ── Step 10: Check ────────────────────────────────────────────────
        $display("----------------------------------------------------------");
        $display(" RESULTS");
        $display("----------------------------------------------------------");
        $display(" best_object = %0d  (expected %0d)",
                  rb_object[15:0], EXPECTED_BEST_OBJECT);
        $display(" best_score  = %0d  (expected %0d)",
                  $signed(rb_score), $signed(EXPECTED_BEST_SCORE));
        $display("----------------------------------------------------------");

        if (rb_object[15:0] == EXPECTED_BEST_OBJECT)
            $display(" [PASS] best_object CORRECT");
        else
            $display(" [FAIL] best_object WRONG - got %0d expected %0d",
                      rb_object[15:0], EXPECTED_BEST_OBJECT);

        if ($signed(rb_score) == $signed(EXPECTED_BEST_SCORE))
            $display(" [PASS] best_score CORRECT");
        else
            $display(" [FAIL] best_score WRONG - got %0d expected %0d",
                      $signed(rb_score), $signed(EXPECTED_BEST_SCORE));

        $display("==========================================================");
        $stop;
    end

    // ─── Global Timeout ───────────────────────────────────────────────────
    initial begin
        #10000000;
        $display("[TB] *** GLOBAL TIMEOUT ***");
        $stop;
    end

endmodule
