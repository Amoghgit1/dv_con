`timescale 1ns / 1ps
// =============================================================================
// max_score.v  — Running maximum tracker
// FIXES:
//   [BUG-C] Reset polarity: changed if(!reset) → if(reset) throughout
// =============================================================================
module max_score (
    input  wire               clk,
    input  wire               reset,          // active-HIGH

    input  wire               i_data_valid,
    input  wire signed [31:0] final_score,
    input  wire        [15:0] object_id,

    output reg                o_data_valid,
    output reg  signed [31:0] best_score,
    output reg         [15:0] best_object
);

    always @(posedge clk) begin
        if (reset) begin                       // [BUG-C] was: if(!reset)
            best_score   <= -32'sd2147483648;  // 32'h8000_0000 — minimum signed 32-bit
            best_object  <= 16'd0;
            o_data_valid <= 1'b0;
        end else begin
            if (i_data_valid) begin
                if (final_score >= best_score) begin
                    best_score  <= final_score;
                    best_object <= object_id;
                end
            end
            o_data_valid <= i_data_valid;
        end
    end

endmodule