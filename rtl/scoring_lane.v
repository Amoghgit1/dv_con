`timescale 1ns / 1ps
// =============================================================
//  scoring_lane.v - Single MAC + Score Pipeline Lane 
//  DVCon India 2026
//
//  Self-contained lane that computes the affinity score for one
//  object.  Encapsulates:
//      MAC  (dot product)  → 1-cycle latency
//      score (clip+boost+confidence → final score)  → 3-cycle latency
//
//  Total pipeline depth: 4 clock cycles from `start` to `done`.
//
//  All inputs must be stable BEFORE `start` is pulsed and must
//  remain stable until `done` fires.
// =============================================================
module scoring_lane #(
    parameter EMB_DIM = 8
)(
    input  wire        clk,
    input  wire        rst,             // active-high synchronous reset

    // ── Control ──────────────────────────────────────────────
    input  wire        start,           // 1-cycle pulse

    // ── Embedding inputs (latched by parent before start) ────
    input  wire signed [8*EMB_DIM-1:0] task_emb_flat,
    input  wire signed [8*EMB_DIM-1:0] label_emb_flat,

    // ── Per-object metadata (latched by parent before start) ─
    input  wire signed [15:0]          confidence,
    input  wire signed [15:0]          boost,
    input  wire                        penalty,

    // ── Outputs ──────────────────────────────────────────────
    output wire signed [31:0]          final_score,
    output wire                        done           // 1-cycle pulse
);

    // ── Internal wires ───────────────────────────────────────
    wire signed [31:0] dot_product;
    wire               mac_done;

    // ── MAC: combinational Σ a[i]*b[i] + 1 register stage ───
    MAC #(.EMB_DIM(EMB_DIM)) u_mac (
        .clk            (clk),
        .reset          (rst),
        .task_emb_flat  (task_emb_flat),
        .label_emb_flat (label_emb_flat),
        .i_data_valid   (start),
        .dot_product    (dot_product),
        .o_data_valid   (mac_done)
    );

    // ── Score: clip_similarity → relevance → final_score ─────
    score u_score (
        .clk          (clk),
        .reset        (rst),
        .dot          (dot_product),
        .confidence   (confidence),
        .boost        (boost),
        .penalty      (penalty),
        .i_data_valid (mac_done),
        .final_score  (final_score),
        .o_data_valid (done)
    );

endmodule