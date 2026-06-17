`timescale 1ns / 1ps
module control#(
    parameter OBJ_CNT_WIDTH = 16,
    parameter EMB_DIM=512
)(
    input clk,
    input reset,
    
    input start,
    input [15:0] n_objects,
    input emd,
    
    input mac_done,
    input score_done,
    input max_done,
    
    output reg  mac_en,
    output reg  score_en,
    output reg  max_en,
    output reg [2:0]state,next,
    output reg [OBJ_CNT_WIDTH-1:0] obj_out,   // expose the counter
    output reg done
    );
    
    
    reg [OBJ_CNT_WIDTH-1:0]obj;
    localparam IDLE= 3'd0,
               LOAD= 3'd1,
               MAC= 3'd2,
               SCORE= 3'd3,
               MAX= 3'd4,
               CHECK= 3'd5;
     always@(posedge clk)
     begin
     if(!reset)begin
        state<=IDLE;
     end else 
        state<=next;
     end
     
 always@(*)begin    
    next=state;
     case(state)
        IDLE: begin
                next= start ? LOAD : IDLE;
            end
        LOAD:begin
               next= emd ? MAC:LOAD;
            end
        MAC:begin
               next= mac_done? SCORE:MAC;
            end
        SCORE:begin
               next= score_done? MAX:SCORE;
            end   
        MAX:begin
               next= max_done? CHECK:MAX;
            end
        CHECK: next = (obj < n_objects - 1) ? LOAD : IDLE;
        
        default: next = IDLE;  
        endcase
      end
    wire enter_mac   = (state != MAC)   && (next == MAC);
    wire enter_score = (state != SCORE) && (next == SCORE);
    wire enter_max   = (state != MAX)   && (next == MAX);

    always @(posedge clk) begin
        if (!reset) begin
            mac_en   <= 1'b0;
            score_en <= 1'b0;
            max_en   <= 1'b0;
        end else begin
            mac_en   <= enter_mac;
            score_en <= enter_score;
            max_en   <= enter_max;
        end
    end
    
       always @(posedge clk) begin
        if (!reset) begin
            obj  <= {OBJ_CNT_WIDTH{1'b0}};
            done <= 1'b0;
        end else begin
            if (state == IDLE && start)
                obj <= {OBJ_CNT_WIDTH{1'b0}};
            else if (state == CHECK && next == LOAD)
                obj <= obj + 1'b1;

            // done is high for exactly one cycle: CHECK -> IDLE transition
            done <= (state == CHECK) && (next == IDLE);
        end
    end
endmodule
