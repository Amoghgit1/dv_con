module embedding_loader #(
    parameter EMB_DIM = 512
)(
    input  wire                          clk,
    input  wire                          reset,    // active LOW sync reset
    input  wire                          clr,       // reset write pointer
    input  wire                          wr_en,     // one pulse per element
    input  wire signed [7:0]             wr_data,
 
    output wire signed [8*EMB_DIM-1:0]   emb_out,
    output reg                           full
);
 
    localparam IDX_WIDTH = $clog2(EMB_DIM);
 
    reg signed [7:0]        buf_mem [0:EMB_DIM-1];
    reg [IDX_WIDTH-1:0]      wr_idx;
 
    integer k;
 
    // ─── Write Pointer + Storage ─────────────────────────────────────────────
    always @(posedge clk) begin
        if (!reset) begin
            wr_idx <= {IDX_WIDTH{1'b0}};
            full   <= 1'b0;
        end
        else begin
            if (clr) begin
                wr_idx <= {IDX_WIDTH{1'b0}};
                full   <= 1'b0;
            end
            else if (wr_en) begin
                buf_mem[wr_idx] <= wr_data;
 
                if (wr_idx == EMB_DIM - 1) begin
                    wr_idx <= {IDX_WIDTH{1'b0}};  // wrap
                    full   <= 1'b1;
                end
                else begin
                    wr_idx <= wr_idx + 1'b1;
                end
            end
        end
    end
 
    // ─── Flatten for top-level bus interface ────────────────────────────────
    genvar gi;
    generate
        for (gi = 0; gi < EMB_DIM; gi = gi + 1) begin : FLATTEN
            assign emb_out[8*gi +: 8] = buf_mem[gi];
        end
    endgenerate
 
endmodule