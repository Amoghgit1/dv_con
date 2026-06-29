`timescale 1ns / 1ps
// =============================================================
//  obj_emb_bram.v - True Dual-Port BRAM for Object Embeddings
//  DVCon India 2026
//
//  Stores one full embedding (EMB_DIM × 8-bit signed bytes) per
//  row.  Inferred as Xilinx Block RAM with byte-write-enable.
//
//  Dimensions:
//    Width = 8 × EMB_DIM bits   (64 bits for EMB_DIM=8)
//    Depth = MAX_OBJECTS entries (16 by default)
//
//  Port A - AXI side:
//    Byte-level writes via one-hot a_we vector.
//    Full-width read (read-first mode).
//
//  Port B - Compute side:
//    Full-width read only.
// =============================================================
module obj_emb_bram #(
    parameter EMB_DIM     = 8,
    parameter MAX_OBJECTS = 16
)(
    input  wire                                clk,

    // ── Port A (AXI side) ────────────────────────────────────
    input  wire [EMB_DIM-1:0]                  a_we,      // byte write enables
    input  wire [$clog2(MAX_OBJECTS)-1:0]      a_addr,
    input  wire [8*EMB_DIM-1:0]                a_din,
    output reg  [8*EMB_DIM-1:0]                a_dout,

    // ── Port B (Compute side - read only) ────────────────────
    input  wire [$clog2(MAX_OBJECTS)-1:0]      b_addr,
    output reg  [8*EMB_DIM-1:0]                b_dout
);

    localparam WIDTH = 8 * EMB_DIM;
    localparam DEPTH = MAX_OBJECTS;

    // ── BRAM inference attribute ─────────────────────────────
    (* ram_style = "block" *) reg [WIDTH-1:0] mem [0:DEPTH-1];

    // ── Initialisation (simulation / bitstream init) ─────────
    integer init_i;
    initial begin
        for (init_i = 0; init_i < DEPTH; init_i = init_i + 1)
            mem[init_i] = {WIDTH{1'b0}};
    end

    // ── Port A: read-first, byte write enable ────────────────
    //    Follows Xilinx UG901 recommended coding template.
    integer j;
    always @(posedge clk) begin
        a_dout <= mem[a_addr];                        // read-first
        for (j = 0; j < EMB_DIM; j = j + 1) begin
            if (a_we[j])
                mem[a_addr][j*8 +: 8] <= a_din[j*8 +: 8];
        end
    end

    // ── Port B: read only ────────────────────────────────────
    always @(posedge clk) begin
        b_dout <= mem[b_addr];
    end

endmodule