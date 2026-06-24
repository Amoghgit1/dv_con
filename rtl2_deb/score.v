`timescale 1ns / 1ps
// =============================================================================
// score.v  — Score computation pipeline
//   Pipeline stages (each registered):
//     Stage 1: clip_score  = (dot + 1) >> 1
//     Stage 2: relevance   = clamp(clip_score + boost, 0, 1000)
//     Stage 3: final_score = (650*relevance + 350*confidence) / 1000
//
// FIXES:
//   [BUG-C] Reset polarity: changed if(!reset) → if(reset) throughout
//   [BUG-D] clip_data_valid was assigned OUTSIDE the if(!reset) branch in stage-1
//           always block, so it was never cleared on reset → spurious valid pulses.
//           Moved inside the reset branch (cleared to 0) and the else branch.
//   [BUG-G] Same problem for rel_data_valid in the stage-2 always block.
//           Moved inside reset / else branches.
// =============================================================================
module score (
    input  wire               clk,
    input  wire               reset,          // active-HIGH

    input  wire signed [31:0] dot,
    input  wire signed [15:0] confidence,
    input  wire signed [15:0] boost,
    input  wire               penalty,

    input  wire               i_data_valid,

    output reg  signed [31:0] final_score,
    output reg                o_data_valid
);

    // ── Stage 1: clip / scale ───────────────────────────────────────────────
    reg signed [31:0] clip_score;
    reg               clip_data_valid;

    always @(posedge clk) begin
        if (reset) begin                    // [BUG-C] was: if(!reset)
            clip_score      <= 32'sd0;
            clip_data_valid <= 1'b0;        // [BUG-D] was outside this block → never reset
        end else begin
            clip_score      <= (dot + 1) >>> 1;   // arithmetic right-shift
            clip_data_valid <= i_data_valid;       // [BUG-D] moved inside else
        end
    end

    // ── Stage 2: relevance with optional penalty and clamp ─────────────────
    reg signed [31:0] relevance_temp;
    reg signed [31:0] relevance;
    reg               rel_data_valid;

    // Combinational: apply penalty (halve) before clamping
    always @(*) begin
        if (penalty)
            relevance_temp = (clip_score + boost) >>> 1;
        else
            relevance_temp = clip_score + boost;
    end

    always @(posedge clk) begin
        if (reset) begin                    // [BUG-C] was: if(!reset)
            relevance      <= 32'sd0;
            rel_data_valid <= 1'b0;         // [BUG-G] was outside this block → never reset
        end else begin
            // Clamp to [0, 1000]
            if      (relevance_temp > 32'sd1000) relevance <= 32'sd1000;
            else if (relevance_temp < 32'sd0)    relevance <= 32'sd0;
            else                                  relevance <= relevance_temp;

            rel_data_valid <= clip_data_valid;    // [BUG-G] moved inside else
        end
    end

    // ── Stage 3: weighted combination ──────────────────────────────────────
    always @(posedge clk) begin
        if (reset) begin                    // [BUG-C] was: if(!reset)
            final_score  <= 32'sd0;
            o_data_valid <= 1'b0;
        end else begin
            final_score  <= (32'sd650 * relevance + 32'sd350 * confidence) / 32'sd1000;
            o_data_valid <= rel_data_valid;
        end
    end

endmodule