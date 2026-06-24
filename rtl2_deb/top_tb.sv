`timescale 1ns / 1ps
/**
 * =============================================================================
 * top_tb.sv - File-Based AXI-Stream Testbench
 * =============================================================================
 * DVCon India 2026 - Stage 2B
 *
 * FIXES applied in this file:
 *   [BUG-TB1] Double $fclose on file_result: the handle was closed at line ~207
 *             and then closed a second time at $stop. Removed the duplicate call.
 *
 * Required files in Vivado project directory:
 *   task_emb.hex    - task embedding (EMB_DIM bytes, one per line, hex)
 *   obj_emb_0.hex   - object 0 embedding
 *   obj_emb_1.hex   - object 1 embedding
 *
 * Golden Reference (computed by export_test_vectors.py):
 *   Expected best_object = 0   (object 0 wins)
 *   Expected best_score  = see Python output
 *
 * Tool : Vivado 2025.1 (XSIM)
 * =============================================================================
 */

module top_tb_sv;

    localparam EMB_DIM   = 8;
    localparam CLK_HALF  = 5;   // 100 MHz

    // ─── Golden Reference ─────────────────────────────────────────────────
    // UPDATE these after running export_test_vectors.py
    localparam EXPECTED_BEST_OBJECT = 16'd0;
    localparam EXPECTED_BEST_SCORE  = 32'd11749;

    // ─── DUT Signals ──────────────────────────────────────────────────────
    reg        clk;
    reg        reset;

    reg        i_data_valid;
    reg  [7:0] i_data;
    wire       o_data_ready;

    wire       o_data_valid;
    wire [7:0] o_data;

    wire       intr;

    // ─── File Handles ──────────────────────────────────────────────────────
    integer    file_result;
    integer    i;

    // ─── Embedding Storage (read from hex files) ───────────────────────────
    reg [7:0] task_emb  [0:EMB_DIM-1];
    reg [7:0] obj_emb_0 [0:EMB_DIM-1];
    reg [7:0] obj_emb_1 [0:EMB_DIM-1];

    // ─── Result Collection ─────────────────────────────────────────────────
    reg  [7:0]  rx_buf [0:5];
    integer     rx_count;
    reg  [15:0] rx_best_object;
    reg  [31:0] rx_best_score;

    // ─── DUT Instantiation ────────────────────────────────────────────────
    affinity_scorer_top #(.EMB_DIM(EMB_DIM)) dut (
        .axi_clk    (clk),
        .axi_reset_n(reset),
        .i_data_valid(i_data_valid),
        .i_data      (i_data),
        .o_data_ready(o_data_ready),
        .o_data_valid(o_data_valid),
        .o_data      (o_data),
        .i_data_ready(1'b1),
        .o_intr      (intr)
    );

    // ─── Clock ────────────────────────────────────────────────────────────
    initial clk = 1'b0;
    always #CLK_HALF clk = ~clk;

    // ─── Waveform Dump ────────────────────────────────────────────────────
    initial begin
        $dumpfile("affinity_scorer_tb.vcd");
        $dumpvars(0, top_tb_sv);
    end

    // ─── Receive Output Bytes ─────────────────────────────────────────────
    // Collect 6 bytes from master stream when they arrive
    initial rx_count = 0;
    always @(posedge clk) begin
        if (o_data_valid && rx_count < 6) begin
            rx_buf[rx_count] = o_data;
            rx_count = rx_count + 1;
        end
    end

    // ─── Helper: Send One Byte ────────────────────────────────────────────
    task send_byte;
        input [7:0] data;
        begin
            @(posedge clk);
            i_data       <= data;
            i_data_valid <= 1'b1;
            @(posedge clk);
            i_data_valid <= 1'b0;
        end
    endtask

    // ─── Helper: Stream Embedding ─────────────────────────────────────────
    task send_embedding;
        input [7:0] type_byte;
        input [7:0] vec [0:EMB_DIM-1];
        integer k;
        begin
            send_byte(type_byte);
            for (k = 0; k < EMB_DIM; k = k + 1)
                send_byte(vec[k]);
        end
    endtask

    // ─── Helper: Send META ────────────────────────────────────────────────
    task send_meta;
        input signed [15:0] conf;
        input signed [15:0] boost;
        input               penalty;
        begin
            send_byte(8'h03);
            send_byte(conf[15:8]);
            send_byte(conf[7:0]);
            send_byte(boost[15:8]);
            send_byte(boost[7:0]);
            send_byte({7'd0, penalty});
        end
    endtask

    // ─── Main Stimulus ────────────────────────────────────────────────────
    initial begin
        // Initialise
        reset        = 1'b0;
        i_data_valid = 1'b0;
        i_data       = 8'h00;

        // Load embedding vectors from hex files
        $readmemh("task_emb.hex",  task_emb);
        $readmemh("obj_emb_0.hex", obj_emb_0);
        $readmemh("obj_emb_1.hex", obj_emb_1);

        // Open result output file
        file_result = $fopen("result.txt", "w");

        // Reset sequence
        #100;
        reset = 1'b1;
        #100;

        $display("==========================================================");
        $display(" DVCon India 2026 - Stage 2B - Affinity Scorer Testbench");
        $display("==========================================================");

        // ── Send task embedding ───────────────────────────────────────────
        $display("[TB] Sending task embedding...");
        send_embedding(8'h01, task_emb);

        // ── Send n_objects = 2 ───────────────────────────────────────────
        $display("[TB] Sending n_objects = 2...");
        send_byte(8'h04);
        send_byte(8'h00);
        send_byte(8'h02);

        // ── Send object 0 embedding + metadata ───────────────────────────
        $display("[TB] Sending object 0 embedding...");
        send_embedding(8'h02, obj_emb_0);
        $display("[TB] Sending object 0 metadata...");
        send_meta(16'sd13107, 16'sd4096, 1'b0);

        // ── START ─────────────────────────────────────────────────────────
        $display("[TB] Sending START...");
        send_byte(8'h05);

        // ── Wait for first intr (object 0 processed) ─────────────────────
        @(posedge intr);
        $display("[TB] intr received - object 0 processed.");

        // ── Send object 1 embedding + metadata ───────────────────────────
        $display("[TB] Sending object 1 embedding...");
        send_embedding(8'h02, obj_emb_1);
        $display("[TB] Sending object 1 metadata...");
        send_meta(16'sd11796, 16'sd1638, 1'b0);

        // ── Wait for final intr (all objects done) ────────────────────────
        @(posedge intr);
        $display("[TB] Final intr received - all objects processed.");

        // ── Wait for all 6 output bytes to be collected ───────────────────
        wait (rx_count == 6);
        repeat(5) @(posedge clk);

        // ── Assemble result ───────────────────────────────────────────────
        rx_best_object = {rx_buf[0], rx_buf[1]};
        rx_best_score  = {rx_buf[2], rx_buf[3], rx_buf[4], rx_buf[5]};

        // ── Write to result file ──────────────────────────────────────────
        $fwrite(file_result, "best_object = %0d\n", rx_best_object);
        $fwrite(file_result, "best_score  = %0d\n", $signed(rx_best_score));
        $fclose(file_result);   // [BUG-TB1] close once only; removed duplicate below

        // ── Display and check ─────────────────────────────────────────────
        $display("----------------------------------------------------------");
        $display(" RESULTS");
        $display("----------------------------------------------------------");
        $display(" best_object = %0d  (expected %0d)",
                  rx_best_object, EXPECTED_BEST_OBJECT);
        $display(" best_score  = %0d  (expected %0d)",
                  $signed(rx_best_score), $signed(EXPECTED_BEST_SCORE));
        $display("----------------------------------------------------------");

        if (rx_best_object == EXPECTED_BEST_OBJECT)
            $display(" [PASS] best_object CORRECT");
        else
            $display(" [FAIL] best_object WRONG");

        if ($signed(rx_best_score) == $signed(EXPECTED_BEST_SCORE))
            $display(" [PASS] best_score CORRECT");
        else
            $display(" [FAIL] best_score WRONG");

        $display("==========================================================");
        // [BUG-TB1] removed second $fclose(file_result) that was here
        $stop;
    end

    // ─── Global Timeout Watchdog ──────────────────────────────────────────
    initial begin
        #1000000;
        $display("[TB] *** GLOBAL TIMEOUT ***");
        $stop;
    end

endmodule