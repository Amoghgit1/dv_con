`timescale 1ns / 1ps
// =============================================================================
// MAC.v  — Multiply-Accumulate (dot product)
// FIXES:
//   [BUG-A] Port changed from unpacked array to flat packed bus to match top-level
//           (was: input signed [7:0] task_emb [0:EMB_DIM-1] — SystemVerilog only,
//            and incompatible with the packed flat wire driven by top)
//   [BUG-B] EMB_DIM parameter was missing at instantiation (default was 512);
//           parameter is now correctly received from top via #(.EMB_DIM(EMB_DIM))
//   [BUG-C] Reset polarity: top drives .reset(rst) which is active-HIGH;
//           changed if(!reset) → if(reset) throughout
// =============================================================================
module MAC #(
    parameter EMB_DIM = 8
)(
    input  wire        clk,
    input  wire        reset,          // active-HIGH (driven by rst in top)

    // Flat packed buses — matches task_emb_flat / label_emb_flat in top
    input  wire signed [8*EMB_DIM-1:0] task_emb_flat,
    input  wire signed [8*EMB_DIM-1:0] label_emb_flat,

    input  wire        i_data_valid,

    output reg  signed [31:0] dot_product,
    output reg                o_data_valid
);

    integer i;
    reg signed [39:0] sum;

    // Unpack flat buses into local arrays for the loop
    wire signed [7:0] task_emb  [0:EMB_DIM-1];
    wire signed [7:0] label_emb [0:EMB_DIM-1];

    genvar gi;
    generate
        for (gi = 0; gi < EMB_DIM; gi = gi + 1) begin : UNPACK
            assign task_emb [gi] = task_emb_flat [8*gi +: 8];
            assign label_emb[gi] = label_emb_flat[8*gi +: 8];
        end
    endgenerate

    // Combinational dot-product accumulation
    always @(*) begin
        sum = 40'sd0;
        for (i = 0; i < EMB_DIM; i = i + 1)
            sum = sum + task_emb[i] * label_emb[i];
    end

    // Register result and valid flag
    always @(posedge clk) begin
        if (reset) begin           // [BUG-C] was: if(!reset)
            dot_product  <= 32'sd0;
            o_data_valid <= 1'b0;
        end else begin
            dot_product  <= sum[31:0];
            o_data_valid <= i_data_valid;
        end
    end

endmodule