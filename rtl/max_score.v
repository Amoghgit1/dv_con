
module max_score(
    input clk,
    input reset,
    input i_data_valid,
    input signed [31:0] final_score,
    input [15:0]object_id,
    
    output reg o_data_valid,
    output reg signed [31:0] best_score,
    output reg [15:0] best_object
    );
    always@(posedge clk)begin
        if(!reset)begin
            best_score   <= -32'sd2147483648;
            best_object<=0;
            o_data_valid <= 0;
        end
        else begin
            if(i_data_valid)begin
               if(final_score>=best_score)begin
                best_score<=final_score;
                best_object <= object_id;
             end
            end
         o_data_valid <= i_data_valid;
         end
        end      
             
endmodule
