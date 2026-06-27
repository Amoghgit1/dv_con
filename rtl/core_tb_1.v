`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 25.06.2026 21:01:52
// Design Name: 
// Module Name: core_tb_1
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


module core_tb_1();
    parameter EMB_DIM = 8;

reg clk;
reg axi_reset_n;

reg start_reg;
reg [15:0] n_objects_reg;

reg signed [15:0] confidence_reg;
reg signed [15:0] boost_reg;
reg penalty_reg;

wire done;
wire [15:0] best_object;
wire signed [31:0] best_score;

integer k;
reg obj1_loaded;

//----------------------------------------------------
// Embedding memories
//----------------------------------------------------

reg signed [7:0] task_mem  [0:EMB_DIM-1];
reg signed [7:0] label_mem [0:EMB_DIM-1];
reg signed [7:0] label_mem1 [0:EMB_DIM-1];

wire signed [8*EMB_DIM-1:0] task_emb_flat;
wire signed [8*EMB_DIM-1:0] label_emb_flat;

genvar i;
generate
for(i=0;i<EMB_DIM;i=i+1)
begin
    assign task_emb_flat [8*i +:8] = task_mem[i];
    assign label_emb_flat[8*i +:8] = label_mem[i];
end
endgenerate

//----------------------------------------------------
// DUT
//----------------------------------------------------

affinity_scorer_top #(
    .EMB_DIM(EMB_DIM)
)dut(

    .axi_clk( clk ),
    .axi_reset_n( axi_reset_n ),

    .start_reg( start_reg ),
    .n_objects_reg( n_objects_reg ),

    .task_emb_flat( task_emb_flat ),
    .label_emb_flat( label_emb_flat ),

    .confidence_reg( confidence_reg ),
    .boost_reg( boost_reg ),
    .penalty_reg( penalty_reg ),

    .done( done ),
    .best_object( best_object ),
    .best_score( best_score )
);

//----------------------------------------------------
// Clock
//----------------------------------------------------

initial clk = 0;
always #5 clk = ~clk;

//----------------------------------------------------
// Waveform
//----------------------------------------------------

initial
begin
    $dumpfile("affinity_core_tb.vcd");
    $dumpvars(0,core_tb_1);
end

//----------------------------------------------------
// Reset
//----------------------------------------------------

task reset_dut;
begin

    axi_reset_n = 0;

    start_reg = 0;
    n_objects_reg = 2;
    obj1_loaded = 0;
    

    confidence_reg = 16'sd13107;
    boost_reg      = 16'sd4096;
    penalty_reg    = 0;

    repeat(5) @(posedge clk);

    axi_reset_n = 1;

    repeat(5) @(posedge clk);

end
endtask

//----------------------------------------------------
// Start pulse
//----------------------------------------------------

task start_core;
begin

    @(posedge clk);
    start_reg <= 1;

    @(posedge clk);
    start_reg <= 0;

end
endtask

//----------------------------------------------------
// Monitor
//----------------------------------------------------

always @(posedge clk)
begin

    $display("T=%0t  state=%0d obj=%0d mac_en=%b score_en=%b max_en=%b done=%b",
            $time,
            dut.FSM.state,
            dut.cur_obj_idx,
            dut.mac_en,
            dut.score_en,
            dut.max_en,
            done);

end
always @(posedge clk)
begin

if(dut.mac_done)
begin
$display("DOT=%0d",dut.dot_product);
end

if(dut.score_done)
begin
$display("FINAL SCORE=%0d",dut.final_score);
end

end
//----------------------------------------------------
// Load Object-1 embedding before second MAC
//----------------------------------------------------

always @(posedge clk)
begin
    if (!obj1_loaded &&
        dut.FSM.state == 3'd5 &&      // CHECK state
        dut.cur_obj_idx == 16'd0)
    begin

        $display("Loading Object-1 embedding...");

        for(k=0;k<EMB_DIM;k=k+1)
            label_mem[k] <= label_mem1[k];

        confidence_reg <= 16'sd11796;   // 0.72*16384
        boost_reg      <= 16'sd1638;    // 0.10*16384
        penalty_reg    <= 0;

        obj1_loaded <= 1;

    end
end
//----------------------------------------------------
// Main
//----------------------------------------------------

initial
begin

    $display("-------------------------------------------");
    $display(" Affinity Core Testbench");
    $display("-------------------------------------------");

    $readmemh("task_emb.hex",task_mem);
    $readmemh("obj_emb_0.hex",label_mem);
    $readmemh("obj_emb_1.hex", label_mem1);

    reset_dut();

    $display("Starting accelerator...");

    start_core();

    wait(done);

    repeat(5) @(posedge clk);

    $display("");
    $display("--------------------------------");
$display("RESULTS");
$display("--------------------------------");

$display("Dot Product  = %0d", dut.dot_product);
$display("Final Score  = %0d", dut.final_score);
$display("Best Score   = %0d", best_score);
$display("Best Object  = %0d", best_object);

$display("--------------------------------");

//------------------------------------------------------
// Automatic Verification
//------------------------------------------------------

if (best_object == 16'd1)
    $display("[PASS] Best Object");
else
    $display("[FAIL] Best Object");

if (best_score == 32'sd11749)
    $display("[PASS] Best Score");
else
    $display("[FAIL] Best Score");

// Optional: Check intermediate results too
if (dut.dot_product == -32'sd2438)
    $display("[PASS] Dot Product");
else
    $display("[FAIL] Dot Product");

if (dut.final_score == 32'sd11749)
    $display("[PASS] Final Score");
else
    $display("[FAIL] Final Score");

$display("--------------------------------");
$display("Core Verification Complete");
$display("--------------------------------");

$stop;


end

//----------------------------------------------------
// Timeout
//----------------------------------------------------

initial
begin

    #5000;

    $display("******** TIMEOUT ********");

    $stop;

end

endmodule
