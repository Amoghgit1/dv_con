`timescale 1ns / 1ps
module affinity_scorer_top#(
    parameter EMB_DIM=512
    )(
    input clk,
    input reset,
    input start,
    input emd_signal,
    input [15:0] n_objects,
   
    
    input signed [8*EMB_DIM-1:0] task_emb_flat,
    input signed [8*EMB_DIM-1:0] label_emb_flat,
   
    input signed [15:0] confidence,
    input signed [15:0] boost,
    input penalty,
    
    output done,
    output wire [15:0] cur_obj_idx,
    output wire signed [31:0] best_score,
    output wire [15:0] best_object
    
    );
    wire max_en,score_en,mac_en;
    wire mac_done, score_done ,  max_done;
    
    wire[15:0]object_id;
    
    wire signed [31:0] dot_product;
    wire signed [31:0] final_score;
    
    
    control FSM(
    .clk(clk),
    .reset(reset),
    
    .start(start),
    .n_objects(n_objects),
    .emb_loaded(emd_signal),
    
    .mac_done(mac_done),
    .score_done(score_done),
    .max_done(max_done),
    
    .mac_en(mac_en),
    .score_en(score_en),
    .max_en(max_en),
    
    .obj(cur_obj_idx),
    .done(done));
    
    MAC mac(
    .clk(clk),
    .reset(reset),
    
    
    .task_emb_flat(task_emb_flat),
    .label_emb_flat(label_emb_flat),
    
    .i_data_valid(mac_en),
    
    .dot_product(dot_product),
    .o_data_valid(mac_done));
    
    score Score(
    .clk(clk),
    .reset(reset),
    
    .dot(dot_product),
    .confidence(confidence),      
    .boost(boost),           
    .penalty(penalty), 
    
    .i_data_valid(mac_done),
    
    .final_score(final_score),   
    .o_data_valid(score_done));
    
    max_score MS(
    .clk(clk),
    .reset(reset),
    .i_data_valid(score_done),
    .final_score(final_score),
    .object_id(cur_obj_idx),
    
    .o_data_valid(max_done),
    .best_score(best_score),
    .best_object(best_object));
endmodule
