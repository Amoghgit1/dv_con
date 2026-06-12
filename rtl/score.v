
module score(
    input clk,
    input reset,
    
    input signed [31:0] dot,
    input signed [15:0] confidence,      
    input signed [15:0] boost,           
    input penalty, 
    
    input i_data_valid,
    
    output reg signed [31:0] final_score,   // 0.65*relevance + 0.35*confidence
    output reg o_data_valid   // result is ready
    );
    
    reg [31:0]score;
    reg signed [31:0]clip_score;
    reg signed [31:0]relevance;
    reg signed [31:0]relevance_temp;
    reg clip_data_valid;
    reg rel_data_valid;
    
    always@(posedge clk)begin
        if(!reset)begin
            clip_score<=0;
        end
        else begin
            clip_score<=(dot+1)>>1;
        end
        clip_data_valid<=i_data_valid;
        
    end
    
    always@(*)begin
        if(penalty)begin
            relevance_temp=(clip_score+boost)>>>1;
        end
        else begin
            relevance_temp=(clip_score+boost);
        end
    end
     always@(posedge clk)begin
        if(relevance_temp > 1000)
        relevance <= 1000;
        else if(relevance_temp < 0)
        relevance <= 0;
        else
        relevance <= relevance_temp;
        
        rel_data_valid<=clip_data_valid;
        
     end
     
     always@(posedge clk)begin
        if(!reset)begin
                final_score<=0;
        end
        else begin
            final_score <=(650*relevance +350*confidence)/1000;
        end
        
        o_data_valid<=rel_data_valid;
     end
     
     
        
endmodule
