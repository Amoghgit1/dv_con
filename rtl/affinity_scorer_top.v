`timescale 1ns / 1ps
module affinity_scorer_top #(
    parameter EMB_DIM = 8
)(
    input  wire        axi_clk,
    input  wire        axi_reset_n,

    input  start_reg,
    input [15:0] n_objects_reg,
    
    
    input signed [8*EMB_DIM-1:0] task_emb_flat,
    input signed [8*EMB_DIM-1:0] label_emb_flat,
    
    input signed [15:0] confidence_reg,
    input signed [15:0] boost_reg,
    input penalty_reg,

    output done,
    output [15:0] best_object,
    output signed [31:0] best_score
);

    // ─── Core Compute Submodules ──────────────────────────────────────────
    wire        mac_en, score_en, max_en;
    wire        mac_done, score_done, max_done;
    
    wire [15:0] cur_obj_idx;

    wire signed [31:0] dot_product;
    wire signed [31:0] final_score;
    

    control FSM (
        .clk      (axi_clk),
        .reset    (!axi_reset_n),
        .start    (start_reg),
        .n_objects(n_objects_reg),
        .emd      (1'b1),
        .mac_done (mac_done),
        .score_done(score_done),
        .max_done (max_done),
        .mac_en   (mac_en),
        .score_en (score_en),
        .max_en   (max_en),
        .obj_out  (cur_obj_idx),
        .done     (done)
    );

    MAC mac (
        .clk           (axi_clk),
        .reset         (!axi_reset_n),
        .task_emb_flat (task_emb_flat),
        .label_emb_flat(label_emb_flat),
        .i_data_valid  (mac_en),
        .dot_product   (dot_product),
        .o_data_valid  (mac_done)
    );

    score Score (
        .clk         (axi_clk),
        .reset       (!axi_reset_n),
        .dot         (dot_product),
        .confidence  (confidence_reg),
        .boost       (boost_reg),
        .penalty     (penalty_reg),
        .i_data_valid(mac_done),
        .final_score (final_score),
        .o_data_valid(score_done)
    );

    max_score MS (
        .clk         (axi_clk),
        .reset       (!axi_reset_n),
        .i_data_valid(score_done),
        .final_score (final_score),
        .object_id   (cur_obj_idx),
        .o_data_valid(max_done),
        .best_score  (best_score),
        .clear(start_reg),
        .best_object (best_object)
    );

endmodule
