`timescale 1ns / 1ps
// =============================================================
//  MAC.v - Multiply-Accumulate (Dot Product) Unit
//  DVCon India 2026
//
//  Computes:  dot_product = Σ task_emb[i] * label_emb[i]
//             for i = 0 .. EMB_DIM-1
//
//  Ports use flat (packed) buses for portability.
//  Combinational accumulation + 1-cycle registered output.
//  Active-high synchronous reset.
// =============================================================
module MAC #(
    parameter EMB_DIM = 8
)(
    input  wire        clk,
    input  wire        reset,           // active-high synchronous reset

    input  wire signed [8*EMB_DIM-1:0] task_emb_flat,
    input  wire signed [8*EMB_DIM-1:0] label_emb_flat,

    input  wire        i_data_valid,

    output reg  signed [31:0] dot_product,
    output reg         o_data_valid
);

    // ── Combinational dot-product ────────────────────────────
    integer i;
    reg signed [39:0] sum;

    always @(*) begin
        sum = 40'sd0;
        for (i = 0; i < EMB_DIM; i = i + 1) begin
            sum = sum + $signed(task_emb_flat[i*8 +: 8]) *
                        $signed(label_emb_flat[i*8 +: 8]);
        end
    end

    // ── Registered output (1-cycle latency) ──────────────────
    always @(posedge clk) begin
        if (reset) begin
            dot_product  <= 32'sd0;
            o_data_valid <= 1'b0;
        end else begin
            dot_product  <= sum[31:0];
            o_data_valid <= i_data_valid;
        end
    end

endmodule