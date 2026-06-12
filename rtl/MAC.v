module MAC #(
    parameter EMB_DIM = 512
)(
    input clk,
    input reset,
    
    
    input signed [7:0] task_emb [0:EMB_DIM-1],
    input signed [7:0] label_emb[0:EMB_DIM-1],
    
    input i_data_valid,
    
    output reg signed [31:0] dot_product,
    output reg o_data_valid
);
integer i;
reg signed [39:0] sum;

always@(*)begin
   sum=0;
   for(i=0;i<EMB_DIM;i=i+1)
   begin
   sum=sum+task_emb[i]*label_emb[i];
   end
end

always@(posedge clk)begin
    if(!reset)begin
        dot_product<=0;
        o_data_valid<=0;
     end
     else 
     begin
         dot_product<=sum;
         o_data_valid<=i_data_valid;
     end 
 end 
     
     
endmodule
